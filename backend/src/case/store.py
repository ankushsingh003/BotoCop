"""
Persistent case CRUD.

A case accumulates evidence asynchronously -- a transaction event today,
a call event next week -- so this layer must be backed by real storage,
not in-memory state, or evidence from earlier events is lost between runs.
"""
import logging
from datetime import timedelta, timezone
from typing import Optional, Dict, Any

from backend.src.case.db import get_session
from backend.src.case.models import Case, CaseEvent, CaseStatus, utcnow


def _as_aware(dt):
    """SQLite drops tzinfo on round-trip; Postgres keeps it. Normalize to
    UTC-aware so comparisons work regardless of backend."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

logger = logging.getLogger("case-store")

STALE_AFTER = timedelta(days=14)


def get_open_case_for_entity(entity_id: str) -> Optional[Case]:
    """Return the most recent non-terminal case for this entity, if any."""
    session = get_session()
    try:
        case = (
            session.query(Case)
            .filter(Case.entity_id == entity_id)
            .filter(Case.status.in_([CaseStatus.OPEN.value, CaseStatus.ESCALATED.value]))
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


def get_all_cases_for_entity(entity_id: str) -> list:
    """Return all historical cases stored in DB for this entity (caller phone / account)."""
    session = get_session()
    try:
        cases = (
            session.query(Case)
            .filter(Case.entity_id == entity_id)
            .order_by(Case.opened_at.desc())
            .all()
        )
        return [{"case_id": str(c.case_id), "status": c.status, "risk_score": c.risk_score, "opened_at": c.opened_at} for c in cases]
    except Exception as e:
        logger.error(f"Failed to query historical cases for entity {entity_id}: {e}")
        return []
    finally:
        session.close()


def create_case(entity_id: str) -> Case:
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


def append_event(
    case_id,
    channel: str,
    pipeline_result: Dict[str, Any],
    raw_ref: Optional[str] = None,
) -> CaseEvent:
    session = get_session()
    try:
        event = CaseEvent(
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
    session = get_session()
    try:
        case = session.query(Case).filter(Case.case_id == case_id).first()
        if case is None:
            return None
        events = (
            session.query(CaseEvent)
            .filter(CaseEvent.case_id == case_id)
            .order_by(CaseEvent.created_at)
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
    session = get_session()
    try:
        return (
            session.query(Case)
            .filter(Case.status.in_([CaseStatus.OPEN.value, CaseStatus.ESCALATED.value]))
            .count()
        )
    finally:
        session.close()


def update_case_status(case_id, status: CaseStatus, risk_score: Optional[float] = None):
    session = get_session()
    try:
        case = session.query(Case).filter(Case.case_id == case_id).first()
        if case is None:
            raise ValueError(f"No such case: {case_id}")
        case.status = status.value if isinstance(status, CaseStatus) else status
        if risk_score is not None:
            case.risk_score = risk_score
        session.commit()
        logger.info(f"Case {case_id} updated -> status={case.status}, risk_score={case.risk_score}")
    finally:
        session.close()
