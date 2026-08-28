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
    CLOSED_FRAUD = "closed_fraud"
    CLOSED_CLEARED = "closed_cleared"
    STALE = "stale"


class Case(Base):
    __tablename__ = "cases"

    case_id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    entity_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default=CaseStatus.OPEN.value)
    risk_score = Column(Float, nullable=False, default=0.0)
    opened_at = Column(DateTime, default=utcnow)
    last_event_at = Column(DateTime, default=utcnow)

    events = relationship(
        "CaseEvent", back_populates="case", order_by="CaseEvent.created_at"
    )


class CaseEvent(Base):
    __tablename__ = "case_events"

    event_id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    case_id = Column(GUID(), ForeignKey("cases.case_id"), nullable=False, index=True)
    channel = Column(String, nullable=False)  # "transaction" | "call" (more later)
    source_identifier = Column(String, nullable=True, index=True)  # Caller phone, account ID, or email
    raw_ref = Column(String, nullable=True)   # pointer to raw artifact (txn id, recording url)
    pipeline_result = Column(JSON, nullable=True)  # normalized violations/severities
    created_at = Column(DateTime, default=utcnow)

    case = relationship("Case", back_populates="events")
