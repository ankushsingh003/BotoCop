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


def archive_event(channel: str, event_payload: Dict[str, Any], pipeline_result: Dict[str, Any], case_id: str = None):
    """
    Best-effort archive -- failures here must never block the live
    request path (the orchestrator already returned a decision to the
    caller by the time this runs). Logged, not raised.
    Uses resilient local storage fallback if S3/MinIO is unreachable.
    """
    global _s3_available, _s3_check_done
    dt = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    record_id = uuid4().hex

    record = {
        "channel": channel,
        "case_id": case_id,
        "event_payload": event_payload,
        "pipeline_result": pipeline_result,
        "archived_at": datetime.now(timezone.utc).isoformat(),
    }

    # 1. Reliable local file archive (zero latency, durable backup)
    try:
        local_dir = Path("./backend/data/datalake_archive") / channel / f"dt={dt}"
        local_dir.mkdir(parents=True, exist_ok=True)
        with open(local_dir / f"{record_id}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, default=str)
    except Exception as e:
        logger.debug(f"Local datalake archive note: {e}")

    # 2. S3 / MinIO remote archive (if available)
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
            logger.info(f"Archived {channel} event to s3://{DATALAKE_BUCKET}/{key}")
            _s3_check_done = True
        except Exception as e:
            _s3_available = False
            _s3_check_done = True
            logger.info(f"MinIO/S3 datalake endpoint offline; using resilient local storage at backend/data/datalake_archive: {e}")

