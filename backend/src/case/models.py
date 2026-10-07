"""
Persistence models for the cross-channel case management layer.

A "case" is the unit of fraud investigation: it groups together every
piece of evidence (transaction audits, call audits, and eventually
video/text audits) tied to the same entity over time, so the eval layer
can reason about combinations of signals rather than one event at a time.
"""
import enum
import uuid
from datetime import datetime, timezone

def utcnow():
    return datetime.now(timezone.utc)

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.types import TypeDecorator, CHAR

Base = declarative_base()


class GUID(TypeDecorator):
    """
    Platform-independent UUID column.

    Uses Postgres' native UUID type in production, and falls back to a
    CHAR(36) string for SQLite so the same models work unmodified for
    local development and tests.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return uuid.UUID(str(value))


class CaseStatus(str, enum.Enum):
    OPEN = "open"
    ESCALATED = "escalated"
    PENDING_REVIEW = "pending_review"
    CLOSED_FRAUD = "closed_fraud"
    CLOSED_CLEARED = "closed_cleared"
    STALE = "stale"


class Case(Base):
    """
    Groups events across channels for one customer over time.
    Called by orchestrator to compute cross-channel risk.
    """
    __tablename__ = "cases"

    case_id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    entity_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default=CaseStatus.OPEN.value)
    risk_score = Column(Float, nullable=False, default=0.0)
    opened_at = Column(DateTime, default=utcnow)
    last_event_at = Column(DateTime, default=utcnow)
    version = Column(Float, nullable=False, default=1.0) # Optimistic concurrency version

    events = relationship(
        "CaseEvent", back_populates="case", order_by="CaseEvent.event_time"
    )
    status_history = relationship("CaseStatusHistory", back_populates="case")


class CaseEvent(Base):
    """
    A single piece of evidence (like a call or transaction) inside a case.
    Called by the orchestrator after a channel pipeline finishes.
    """
    __tablename__ = "case_events"

    event_id = Column(String, primary_key=True) # Now provided by caller or derived for idempotency
    case_id = Column(GUID(), ForeignKey("cases.case_id"), nullable=False, index=True)
    channel = Column(String, nullable=False)  # "transaction" | "call" (more later)
    raw_ref = Column(String, nullable=True)   # pointer to raw artifact (txn id, recording url)
    pipeline_result = Column(JSON, nullable=True)  # normalized violations/severities
    event_time = Column(DateTime, nullable=False, default=utcnow)
    created_at = Column(DateTime, default=utcnow)

    case = relationship("Case", back_populates="events")


class EventInbox(Base):
    """
    Idempotency inbox table that claims each incoming event exactly once.
    Called by the orchestrator as the very first step of handling an event.
    """
    __tablename__ = "event_inbox"
    event_id = Column(String, primary_key=True)
    channel = Column(String, nullable=False)
    status = Column(String, nullable=False)   # "processing" | "done"
    case_id = Column(GUID(), nullable=True)
    entity_id = Column(String, nullable=True, index=True) # Used for pending_count fast-lane
    received_at = Column(DateTime, default=utcnow)


class CaseStatusHistory(Base):
    """
    Audit trail for state machine transitions (e.g., OPEN -> ESCALATED).
    Called by the case store whenever a case changes status.
    """
    __tablename__ = "case_status_history"
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    case_id = Column(GUID(), ForeignKey("cases.case_id"), nullable=False, index=True)
    from_status = Column(String, nullable=False)
    to_status = Column(String, nullable=False)
    actor = Column(String, nullable=False) # e.g., "AI_JUDGE", "ANALYST_SMITH"
    reason = Column(String, nullable=True)
    changed_at = Column(DateTime, default=utcnow)

    case = relationship("Case", back_populates="status_history")


class ReviewQueue(Base):
    """
    Persistent queue for human analysts to review escalated cases.
    Called by the notification/action layer to push tasks to analysts.
    """
    __tablename__ = "review_queue"
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    event_id = Column(String, unique=True, nullable=False) # Ensure 1 review item per event
    case_id = Column(GUID(), ForeignKey("cases.case_id"), nullable=False)
    priority = Column(String, nullable=False, default="medium")
    status = Column(String, nullable=False, default="open") # "open", "claimed", "resolved"
    sla_due_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=utcnow)


class Blocklist(Base):
    """
    Persistent blocklist for malicious actors (phone numbers, emails, accounts).
    Called by the fast-lane endpoints to block incoming activity immediately.
    """
    __tablename__ = "blocklist"
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    type = Column(String, nullable=False) # "phone", "email", "account"
    value = Column(String, nullable=False, index=True)
    reason = Column(String, nullable=True)
    added_at = Column(DateTime, default=utcnow)
    expires_at = Column(DateTime, nullable=True)


class DeadLetterQueue(Base):
    """
    Parking lot for broken or malformed messages that crash the system.
    We save them here instead of letting them clog up the pipeline or get lost.
    Called by the orchestrator when an unhandled exception occurs.
    """
    __tablename__ = "dead_letter_queue"
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    channel = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    error_message = Column(String, nullable=False)
    failed_at = Column(DateTime, default=utcnow)


class Outbox(Base):
    """
    Transactional outbox for archiving events to S3/Datalake safely.
    Written in the same transaction as the case event, guaranteeing no data loss.
    """
    __tablename__ = "outbox"
    event_id = Column(String, primary_key=True)
    channel = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    status = Column(String, nullable=False, default="pending") # "pending" | "sent"
    created_at = Column(DateTime, default=utcnow)
    sent_at = Column(DateTime, nullable=True)

class AnalystLabel(Base):
    """
    Records human analyst decisions for ML model training feedback loops.
    """
    __tablename__ = "analyst_labels"
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    event_id = Column(String, index=True, nullable=False)
    case_id = Column(GUID(), index=True, nullable=False)
    label = Column(String, nullable=False) # "fraud", "not_fraud"
    source = Column(String, nullable=False) # "analyst", "customer_dispute"
    labeled_at = Column(DateTime, default=utcnow)
    delay_seconds = Column(Float, nullable=True) # time between event and label
    model_version = Column(String, nullable=True) # what model scored this?

