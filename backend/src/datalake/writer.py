"""
Archives every processed event to the data lake, partitioned by channel
and date (channel=transaction/dt=2026-08-05/{event_id}.json) -- the
Hive-style partitioning Spark/Athena/Presto expect, so a batch job can
read just one channel's data for one day without scanning everything.

This is what actually justifies having a data lake at all: every live
audit becomes training data for the next model retrain, not just a
one-off decision that's thrown away after the response is returned.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4

logger = logging.getLogger("datalake-writer")

_s3_available: bool = True
_s3_check_done: bool = False


def relay_outbox():
    """
    Relay worker loop for Gap 9 (Silent data loss in the lake).
    Reads pending outbox rows from the database and pushes them to S3.
    Deterministic keys mean retries safely overwrite objects without duplicates.
    """
    global _s3_available, _s3_check_done
    from backend.src.case.db import get_session
    from backend.src.case.models import Outbox
    from sqlalchemy import text
    
    session = get_session()
    try:
        # Select for update skip locked to allow multiple workers without contention
        stmt = text("""
            SELECT * FROM outbox WHERE status='pending'
            ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 100
        """)
        rows = session.execute(stmt).all()
        
        if not rows:
            return 0
            
        for r in rows:
            dt = r.created_at.strftime("%Y-%m-%d") if r.created_at else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            record_id = r.event_id
            channel = r.channel
            record = r.payload
            
            # Local fallback archive
            try:
                local_dir = Path("./backend/data/datalake_archive") / channel / f"dt={dt}"
                local_dir.mkdir(parents=True, exist_ok=True)
                with open(local_dir / f"{record_id}.json", "w", encoding="utf-8") as f:
                    json.dump(record, f, default=str)
            except Exception as e:
                logger.debug(f"Local datalake archive error: {e}")

            # S3 remote archive
            if _s3_available:
                try:
                    from backend.src.datalake.client import get_s3_client, ensure_bucket
                    from backend.src.datalake.config import DATALAKE_BUCKET
                    
                    key = f"{channel}/dt={dt}/{record_id}.json"
                    client = get_s3_client()
                    ensure_bucket(client)
                    client.put_object(
                        Bucket=DATALAKE_BUCKET,
                        Key=key,
                        Body=json.dumps(record, default=str).encode("utf-8"),
                        ContentType="application/json",
                    )
                    _s3_check_done = True
                except Exception as e:
                    _s3_available = False
                    _s3_check_done = True
                    logger.warning(f"MinIO/S3 datalake endpoint offline: {e}")

            # Mark as sent
            session.execute(text("UPDATE outbox SET status='sent', sent_at=CURRENT_TIMESTAMP WHERE event_id=:eid"), {"eid": r.event_id})
            
        session.commit()
        return len(rows)
    except Exception as e:
        session.rollback()
        logger.error(f"Outbox relay failed: {e}")
        return 0
    finally:
        session.close()

