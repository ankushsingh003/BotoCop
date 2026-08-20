import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.src.case.models import Base

# Defaults to a local SQLite file so the case layer works out of the box
# with zero setup. Point CASE_DATABASE_URL at Postgres in production,
# e.g. postgresql+psycopg2://user:pass@host:5432/botocop
DATABASE_URL = os.getenv("CASE_DATABASE_URL", "sqlite:///./backend/data/cases.db")

_engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    if ":memory:" in DATABASE_URL:
        # In-memory sqlite normally resets per-connection; pin it to a
        # single connection so state survives across sessions (tests only).
        _engine_kwargs["poolclass"] = StaticPool

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    os.makedirs(os.path.dirname(DATABASE_URL.split("///")[-1]) or ".", exist_ok=True) \
        if DATABASE_URL.startswith("sqlite") and ":memory:" not in DATABASE_URL else None
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()
