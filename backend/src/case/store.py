"""
Persistent case CRUD.

A case accumulates evidence asynchronously -- a transaction event today,
a call event next week -- so this layer must be backed by real storage,
not in-memory state, or evidence from earlier events is lost between runs.
"""
import logging
from datetime import timedelta, timezone
from typing import Optional, Dict, Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy import select, update
from backend.src.case.db import get_session
from backend.src.case.models import Case, CaseEvent, CaseStatus, EventInbox, CaseStatusHistory, utcnow


def _as_aware(dt):
    """
    Helps compare dates by making sure they all have timezones.
    Calls nothing. Used internally by database queries.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

logger = logging.getLogger("case-store")

STALE_AFTER = timedelta(days=14)


def claim_event(event_id: str, channel: str, entity_id: str = None) -> bool:
    """
    Idempotency check: claims an event ID so we never process the same event twice.
    Calls the EventInbox table in the database.
    """
    session = get_session()
    try:
        inbox = session.query(EventInbox).filter_by(event_id=event_id).first()
        if inbox:
            return False
        
        new_claim = EventInbox(event_id=event_id, channel=channel, status="processing", entity_id=entity_id)
        session.add(new_claim)
        session.commit()
        return True
    except IntegrityError:
        session.rollback()
        return False
    finally:
        session.close()


def get_open_case_for_entity(entity_id: str) -> Optional[Case]:
    """
    Finds if this customer already has a fraud case that is currently active.
    Calls the database to check the 'cases' table.
    """
    """Return the most recent non-terminal case for this entity, if any."""
    session = get_session()
    try:
        case = (
            session.query(Case)
            .filter(Case.entity_id == entity_id)
            .filter(Case.status.in_([
                CaseStatus.OPEN.value, 
                CaseStatus.ESCALATED.value,
                CaseStatus.PENDING_REVIEW.value,
            ]))
            .order_by(Case.last_event_at.desc())
            .first()
        )
        if case and (utcnow() - _as_aware(case.last_event_at)) > STALE_AFTER:
            logger.info(f"Case {case.case_id} for entity {entity_id} is stale; opening a fresh one.")
            case.status = CaseStatus.STALE.value
            session.commit()
            return None
        return case
    finally:
        session.close()


def create_case(entity_id: str) -> Case:
    """
    Creates a brand new fraud case for a customer who doesn't have one yet.
    Calls the database to insert a new row in the 'cases' table.
    """
    session = get_session()
    try:
        case = Case(entity_id=entity_id, status=CaseStatus.OPEN.value, risk_score=0.0)
        session.add(case)
        session.commit()
        session.refresh(case)
        logger.info(f"Opened new case {case.case_id} for entity {entity_id}")
        return case
    finally:
        session.close()


def append_and_score(
    case_id: str, 
    channel: str, 
    pipeline_result: Dict[str, Any], 
    raw_ref: Optional[str] = None, 
    event_id: Optional[str] = None, 
    event_time=None,
    event_payload: Dict[str, Any] = None
) -> tuple:
    """
    Safely adds a new event to a case and recalculates the risk score immediately.
    Uses row locks to prevent two events from overwriting each other simultaneously.
    Calls the database to lock the case, append the event, and update the score.
    """
    import uuid
    from backend.src.case.aggregator import compute_case_risk
    
    session = get_session()
    try:
        # Transaction A (short, locked): lock the case row, append event, recompute risk, commit.
        case = session.execute(
            select(Case).where(Case.case_id == case_id).with_for_update()
        ).scalar_one()
        
        actual_event_id = event_id or str(uuid.uuid4())
        event = CaseEvent(
            event_id=actual_event_id,
            case_id=case_id,
            channel=channel,
            raw_ref=raw_ref,
            pipeline_result=pipeline_result,
            event_time=event_time or utcnow()
        )
        session.add(event)
        
        # Gap 9: Outbox pattern for guaranteed datalake archival
        if event_payload is not None:
            from backend.src.case.models import Outbox
            full_record = {
                "channel": channel,
                "case_id": str(case_id),
                "event_payload": event_payload,
                "pipeline_result": pipeline_result,
            }
            outbox = Outbox(
                event_id=actual_event_id,
                channel=channel,
                payload=full_record,
                status="pending"
            )
            session.add(outbox)
            
        session.flush()
        
        # Load all events to recalculate risk
        events = session.query(CaseEvent).filter(CaseEvent.case_id == case_id).order_by(CaseEvent.event_time).all()
        events_dicts = [{"channel": e.channel, "pipeline_result": e.pipeline_result} for e in events]
        
        risk = compute_case_risk(events_dicts)
        case.risk_score = risk["risk_score"]
        case.version += 1
        case.last_event_at = utcnow()
        
        new_version = case.version
        should_escalate = risk["should_escalate"]
        
        session.commit()
        logger.info(f"Appended {channel} event {actual_event_id} to case {case_id} (new risk: {case.risk_score:.2f})")
        return new_version, should_escalate, risk
    finally:
        session.close()


def append_event(
    case_id,
    channel: str,
    pipeline_result: Dict[str, Any],
    raw_ref: Optional[str] = None,
) -> CaseEvent:
    """
    Legacy method to add an event. (To be deprecated in favor of append_and_score).
    Calls the database to add a CaseEvent row.
    """
    session = get_session()
    try:
        import uuid
        event = CaseEvent(
            event_id=str(uuid.uuid4()),
            case_id=case_id,
            channel=channel,
            raw_ref=raw_ref,
            pipeline_result=pipeline_result,
        )
        session.add(event)

        case = session.query(Case).filter(Case.case_id == case_id).first()
        if case is not None:
            case.last_event_at = utcnow()

        session.commit()
        session.refresh(event)
        logger.info(f"Appended {channel} event {event.event_id} to case {case_id}")
        return event
    finally:
        session.close()


def get_case_with_events(case_id) -> Optional[Dict[str, Any]]:
    """
    Gets all the details of a case, including every event (call, transaction) inside it.
    Calls the database to fetch the Case and its CaseEvents.
    """
    session = get_session()
    try:
        case = session.query(Case).filter(Case.case_id == case_id).first()
        if case is None:
            return None
        events = (
            session.query(CaseEvent)
            .filter(CaseEvent.case_id == case_id)
            .order_by(CaseEvent.event_time)
            .all()
        )
        return {
            "case_id": str(case.case_id),
            "entity_id": case.entity_id,
            "status": case.status,
            "risk_score": case.risk_score,
            "opened_at": case.opened_at,
            "last_event_at": case.last_event_at,
            "events": [
                {
                    "event_id": str(e.event_id),
                    "channel": e.channel,
                    "pipeline_result": e.pipeline_result,
                    "created_at": e.created_at,
                }
                for e in events
            ],
        }
    finally:
        session.close()


def count_open_cases() -> int:
    """
    Counts how many fraud cases are currently active and waiting for resolution.
    Calls the database to count cases where status is OPEN, ESCALATED, or PENDING_REVIEW.
    """
    session = get_session()
    try:
        return (
            session.query(Case)
            .filter(Case.status.in_([
                CaseStatus.OPEN.value, 
                CaseStatus.ESCALATED.value,
                CaseStatus.PENDING_REVIEW.value,
            ]))
            .count()
        )
    finally:
        session.close()


def mark_event_done(event_id: str):
    """Marks an event as finished processing in the inbox."""
    session = get_session()
    try:
        session.query(EventInbox).filter_by(event_id=event_id).update({"status": "done"})
        session.commit()
    finally:
        session.close()


def pending_count(entity_id: str) -> int:
    """Returns the number of events currently being processed for this entity."""
    from sqlalchemy import func
    session = get_session()
    try:
        return session.scalar(
            select(func.count())
            .select_from(EventInbox)
            .where(EventInbox.entity_id == entity_id, EventInbox.status == "processing")
        )
    finally:
        session.close()


ALLOWED_TRANSITIONS = {
  "open":           {"escalated", "closed_cleared", "stale", "pending_review"},
  "pending_review": {"escalated", "closed_cleared", "closed_fraud"},
  "escalated":      {"closed_fraud", "closed_cleared"},
  "closed_fraud":   set(), 
  "closed_cleared": set(), 
  "stale": set(),
}

def transition(session, case, new_status, actor, reason):
    """
    State machine helper: safely moves a case from one status to another and records the history.
    Calls the database to save a CaseStatusHistory record.
    """
    if new_status not in ALLOWED_TRANSITIONS.get(case.status, set()):
        logger.warning(f"Invalid transition from {case.status} to {new_status} for case {case.case_id}")
        return # Ignore invalid transitions gracefully
        
    session.add(CaseStatusHistory(
        case_id=case.case_id, 
        from_status=case.status,
        to_status=new_status, 
        actor=actor, 
        reason=reason
    ))
    case.status = new_status


def apply_decision(case_id: str, seen_version: float, new_status: str, actor: str = "AI_JUDGE", reason: str = "") -> bool:
    """
    Optimistic Concurrency Control (OCC): Applies the LLM's decision ONLY if the case hasn't changed.
    Calls the database to update the case status based on version checking.
    """
    session = get_session()
    try:
        case = session.query(Case).filter(Case.case_id == case_id).first()
        if not case or case.version != seen_version:
            return False # Case changed meanwhile, need to re-evaluate
            
        transition(session, case, new_status, actor, reason)
        case.version += 1
        session.commit()
        return True
    finally:
        session.close()


def update_case_status(case_id, status: CaseStatus, risk_score: Optional[float] = None):
    """
    Legacy method to update status (To be deprecated in favor of apply_decision).
    Calls the database to update the status column.
    """
    session = get_session()
    try:
        case = session.query(Case).filter(Case.case_id == case_id).first()
        if case is None:
            raise ValueError(f"No such case: {case_id}")
            
        new_status = status.value if isinstance(status, CaseStatus) else status
        transition(session, case, new_status, "LEGACY_UPDATE", "Legacy status update")
        
        if risk_score is not None:
            case.risk_score = risk_score
        session.commit()
        logger.info(f"Case {case_id} updated -> status={case.status}, risk_score={case.risk_score}")
    finally:
        session.close()


def resolve_case(case_id: str, new_status: str, actor: str, reason: str, blocklist_entries: list = None) -> bool:
    """
    Finalizes a fraud case (closing it as fraud or cleared) and adds the attacker to the blocklist.
    Calls the database to update Case, CaseStatusHistory, and Blocklist tables in ONE locked transaction.
    Also creates AnalystLabels for ML model feedback loops.
    """
    from backend.src.case.models import Blocklist, AnalystLabel, CaseEvent
    import time
    session = get_session()
    try:
        case = session.query(Case).filter(Case.case_id == case_id).with_for_update().first()
        if not case:
            return False
            
        transition(session, case, new_status, actor, reason)
        case.version += 1
        
        # Determine ML label based on resolution
        ml_label = "fraud" if new_status == "closed_fraud" else "not_fraud"
        
        # Create labels for all events in this case
        events = session.query(CaseEvent).filter(CaseEvent.case_id == case_id).all()
        now_dt = utcnow()
        for event in events:
            delay = (now_dt - event.event_time).total_seconds() if getattr(event, "event_time", None) else None
            ml_score_data = (event.pipeline_result or {}).get("ml_score", {})
            model_version = ml_score_data.get("model_version", "v1.0.0") if isinstance(ml_score_data, dict) else "v1.0.0"
            
            label = AnalystLabel(
                event_id=event.event_id,
                case_id=case_id,
                label=ml_label,
                source=actor,
                delay_seconds=delay,
                model_version=model_version
            )
            session.add(label)
        
        if blocklist_entries and new_status == "closed_fraud":
            for entry in blocklist_entries:
                # Upsert into blocklist
                stmt = select(Blocklist).where(Blocklist.type == entry['type'], Blocklist.value == entry['value'])
                existing = session.execute(stmt).scalar_one_or_none()
                if not existing:
                    bl_entry = Blocklist(type=entry['type'], value=entry['value'], reason=reason)
                    session.add(bl_entry)

        session.commit()
        logger.info(f"Case {case_id} resolved as {new_status} by {actor}")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to resolve case {case_id}: {e}")
        raise
    finally:
        session.close()


def get_recent_cases(limit: int = 30) -> list:
    """
    Retrieves a list of the most recently updated fraud cases for the dashboard.
    Calls the database to fetch cases sorted by latest event time.
    """
    session = get_session()
    try:
        cases = (
            session.query(Case)
            .order_by(Case.last_event_at.desc().nullslast(), Case.opened_at.desc())
            .limit(limit)
            .all()
        )
        results = []
        for c in cases:
            latest_event = (
                session.query(CaseEvent)
                .filter(CaseEvent.case_id == c.case_id)
                .order_by(CaseEvent.created_at.desc())
                .first()
            )
            event_count = session.query(CaseEvent).filter(CaseEvent.case_id == c.case_id).count()
            results.append({
                "case_id": str(c.case_id),
                "entity_id": c.entity_id,
                "status": c.status,
                "risk_score": round(c.risk_score or 0.0, 3),
                "opened_at": c.opened_at.isoformat() if c.opened_at else None,
                "last_event_at": c.last_event_at.isoformat() if c.last_event_at else None,
                "event_count": event_count,
                "latest_channel": latest_event.channel if latest_event else "unknown",
                "latest_status": (latest_event.pipeline_result or {}).get("final_status") if latest_event else "unknown",
                "violations_count": len((latest_event.pipeline_result or {}).get("violations", [])) if latest_event else 0,
            })
        return results
    finally:
        session.close()


def get_case_statistics() -> dict:
    """
    Calculates summary numbers like total cases and open alerts for the dashboard.
    Calls the database to count rows in the cases and case_events tables.
    """
    session = get_session()
    try:
        total_cases = session.query(Case).count()
        open_cases = session.query(Case).filter(Case.status == CaseStatus.OPEN.value).count()
        escalated_cases = session.query(Case).filter(Case.status == CaseStatus.ESCALATED.value).count()
        blocked_cases = session.query(Case).filter(Case.status == CaseStatus.CLOSED_FRAUD.value).count()
        total_events = session.query(CaseEvent).count()
        high_risk_cases = session.query(Case).filter(Case.risk_score >= 0.5).count()
        return {
            "total_cases": total_cases,
            "open_cases": open_cases,
            "escalated_cases": escalated_cases,
            "blocked_cases": blocked_cases,
            "total_events": total_events,
            "high_risk_cases": high_risk_cases,
        }
    finally:
        session.close()


def save_to_dlq(channel: str, payload: dict, error_message: str):
    """
    Saves a broken message that crashed the system into the Dead Letter Queue.
    This ensures we don't lose the data, but it also doesn't stop other messages.
    Calls the database to insert a new row in the 'dead_letter_queue' table.
    """
    from backend.src.case.models import DeadLetterQueue
    session = get_session()
    try:
        dlq_entry = DeadLetterQueue(
            channel=channel,
            payload=payload,
            error_message=error_message
        )
        session.add(dlq_entry)
        session.commit()
        logger.info(f"Saved poison message to DLQ: {channel}")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save to DLQ! {e}")
    finally:
        session.close()


