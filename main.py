import time
import logging
import json
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from dotenv import load_dotenv


from sqlalchemy import create_engine, Column, String, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv(override=True)
from backend.src.orchestrator.orchestrator import handle_event
from backend.src.case.db import init_db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db-ingestion")

INGESTION_DB_URL = os.getenv("INGESTION_DB_URL", "sqlite:///./backend/data/legacy_events.db")
engine = create_engine(INGESTION_DB_URL, connect_args={"check_same_thread": False} if "sqlite" in INGESTION_DB_URL else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class LegacyEvent(Base):
    __tablename__ = "legacy_events"
    
    id = Column(String, primary_key=True)
    event_type = Column(String, nullable=False) # 'transaction', 'call', 'text'
    payload = Column(Text, nullable=False) # JSON string
    processed = Column(Boolean, default=False)

def seed_database():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    
    if session.query(LegacyEvent).count() == 0:
        logger.info("Seeding legacy database with initial events...")
        events = [
            LegacyEvent(
                id="txn_9012",
                event_type="transaction",
                payload=json.dumps({
                    "id": "txn_9012",
                    "customer_id": "cust_555",
                    "amount": 4500.00,
                    "ip_address": "192.168.1.100",
                    "merchant": "HighRisk Electronics"
                })
            ),
            LegacyEvent(
                id="call_334",
                event_type="call",
                payload=json.dumps({
                    "id": "call_334",
                    "customer_id": "cust_555",
                    "transcript": "Hello I need to reset my password and change my shipping address.",
                    "duration_seconds": 120
                })
            )
        ]
        session.add_all(events)
        session.commit()
    session.close()

def run_db_polling_simulation():
    """
    Polls the legacy SQL database for unprocessed events and pushes them
    into the fraud detection orchestrator.
    """
    logger.info("Starting SQL Database Polling...")
    
    init_db()
    
    seed_database()
    
    while True:
        session = SessionLocal()
        try:
            unprocessed = session.query(LegacyEvent).filter(LegacyEvent.processed == False).first()
            
            if unprocessed:
                logger.info(f"Polled new record: {unprocessed.id} (Type: {unprocessed.event_type})")
                
                payload = json.loads(unprocessed.payload)
                
                try:
                    result = handle_event(unprocessed.event_type, payload)
                    logger.info(f"Successfully processed {unprocessed.id}. Case ID: {result['case_id']} | Risk Score: {result.get('case_risk', {}).get('risk_score')}")
                    
                    unprocessed.processed = True
                    session.commit()
                except Exception as e:
                    logger.error(f"Failed to process row {unprocessed.id}: {e}")
                    unprocessed.processed = True
                    session.commit()
                    
            else:
                logger.info("All events processed. Resetting legacy event queue for continuous simulation...")
                session.query(LegacyEvent).update({LegacyEvent.processed: False})
                session.commit()
                
        finally:
            session.close()
            
        time.sleep(3) # Poll every 3 seconds


if __name__ == "__main__":
    run_db_polling_simulation()
