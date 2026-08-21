import time
import logging
from dotenv import load_dotenv

load_dotenv(override=True)
from backend.src.orchestrator.orchestrator import handle_event

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db-ingestion")

def run_db_polling_simulation():
    """
    Simulates a polling process that reads from a legacy SQL database
    (e.g. tracking customer orders or support tickets) and pushes them
    into the fraud detection orchestrator.
    """
    logger.info("Starting SQL Database Polling Simulation...")
    
    # Simulated rows from a legacy database (e.g. SELECT * FROM events WHERE processed = false)
    db_rows = [
        {
            "id": "txn_9012",
            "type": "transaction",
            "customer_id": "cust_555",
            "amount": 4500.00,
            "ip_address": "192.168.1.100",
            "merchant": "HighRisk Electronics"
        },
        {
            "id": "call_334",
            "type": "call",
            "customer_id": "cust_555",
            "transcript": "Hello I need to reset my password and change my shipping address.",
            "duration_seconds": 120
        }
    ]
    
    for row in db_rows:
        logger.info(f"Polled new record: {row['id']} (Type: {row['type']})")
        channel = row.pop("type")
        
        try:
            result = handle_event(channel, row)
            logger.info(f"Successfully processed {row['id']}. Case ID: {result['case_id']} | Risk Score: {result['case_risk']['risk_score']}")
        except Exception as e:
            logger.error(f"Failed to process row {row['id']}: {e}")
            
        time.sleep(2) # Simulate polling delay

if __name__ == "__main__":
    run_db_polling_simulation()
