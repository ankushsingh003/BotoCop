import os
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

    # Clear legacy table to re-seed with rich 5-Layer Call Fraud events
    session.query(LegacyEvent).delete()
    session.commit()

    logger.info("Seeding database with comprehensive 5-Layer Call & Multi-Channel Fraud Events for Grafana monitoring...")
    events = [
        # Event 1: Normal Mobile Call
        LegacyEvent(
            id="call_normal_101",
            event_type="call",
            payload=json.dumps({
                "id": "call_normal_101",
                "caller_phone": "+919811100011",
                "linked_account_id": "cust_101",
                "transcript": "Hello, I am calling to confirm my doctor appointment for tomorrow at 10 AM.",
                "duration_seconds": 35,
                "stir_shaken_attestation": "A",
                "line_type": "MOBILE",
                "hour_of_day": 14,
            })
        ),
        # Event 2: Hinglish Vishing Digital Arrest Scam Call
        LegacyEvent(
            id="call_vishing_scam_102",
            event_type="call",
            payload=json.dumps({
                "id": "call_vishing_scam_102",
                "caller_phone": "+919777888999",
                "linked_account_id": "cust_102",
                "transcript": "Namaste. Main Mumbai Police Cyber Cell se Inspector Sharma bol raha hu. Aapke name par legal warrant hai. Khata band ho jayega, abhi paisa bhejo.",
                "duration_seconds": 120,
                "stir_shaken_attestation": "C",
                "line_type": "NON_FIXED_VOIP",
                "hour_of_day": 23,
                "complaint_history_count": 4,
            })
        ),
        # Event 3: High-Velocity Boiler Room Fan-Out Call 1
        LegacyEvent(
            id="call_boiler_room_103",
            event_type="call",
            payload=json.dumps({
                "id": "call_boiler_room_103",
                "caller_phone": "+919999000888",
                "linked_account_id": "cust_103",
                "transcript": "Urgent alert: HDFC Bank security check. Share your OTP code immediately.",
                "duration_seconds": 45,
                "stir_shaken_attestation": "C",
                "line_type": "NON_FIXED_VOIP",
                "hour_of_day": 22,
            })
        ),
        # Event 4: High-Velocity Boiler Room Fan-Out Call 2 (Same scammer, different target)
        LegacyEvent(
            id="call_boiler_room_104",
            event_type="call",
            payload=json.dumps({
                "id": "call_boiler_room_104",
                "caller_phone": "+919999000888",
                "linked_account_id": "cust_104",
                "transcript": "Urgent alert: SBI account suspended. Give OTP right now.",
                "duration_seconds": 50,
                "stir_shaken_attestation": "C",
                "line_type": "NON_FIXED_VOIP",
                "hour_of_day": 22,
            })
        ),
        # Event 5: Known Scam Blocklist Number Retrying (Sub-millisecond Short-Circuit)
        LegacyEvent(
            id="call_blocklist_retry_105",
            event_type="call",
            payload=json.dumps({
                "id": "call_blocklist_retry_105",
                "caller_phone": "+919876543210",  # Pre-seeded I4C blocklisted number
                "linked_account_id": "cust_105",
                "transcript": "Hello, this is customer care calling.",
                "duration_seconds": 15,
            })
        ),
        # Event 6: High Risk Transaction Fraud
        LegacyEvent(
            id="txn_high_risk_106",
            event_type="transaction",
            payload=json.dumps({
                "id": "txn_high_risk_106",
                "customer_id": "cust_102",
                "amount": 95000.00,
                "ip_address": "103.45.12.8",
                "merchant": "Crypto Exchange LLC",
            })
        ),
    ]
    session.add_all(events)
    session.commit()
    session.close()
    logger.info(f"Database seeded with {len(events)} events for continuous orchestrator pipeline simulation.")


import threading
import uvicorn
from backend.src.api.server import app

import socket

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_metrics_server():
    if is_port_in_use(8000):
        logger.info("Prometheus metrics server is already active on port 8000.")
        return
    try:
        logger.info("Starting Prometheus metrics API server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    except Exception as e:
        logger.warning(f"Metrics server start notice: {e}")


def run_db_polling_simulation():
    """
    Polls the legacy SQL database for unprocessed events and pushes them
    into the fraud detection orchestrator.
    """
    threading.Thread(target=start_metrics_server, daemon=True).start()

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
