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
from typing import Dict, Any
from uuid import uuid4

from backend.src.datalake.client import get_s3_client, ensure_bucket
from backend.src.datalake.config import DATALAKE_BUCKET

logger = logging.getLogger("datalake-writer")


def archive_event(channel: str, event_payload: Dict[str, Any], pipeline_result: Dict[str, Any], case_id: str = None):
    """
    Best-effort archive -- failures here must never block the live
    request path (the orchestrator already returned a decision to the
    caller by the time this runs). Logged, not raised.
    """
    dt = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"{channel}/dt={dt}/{uuid4().hex}.json"

    record = {
        "channel": channel,
        "case_id": case_id,
        "event_payload": event_payload,
        "pipeline_result": pipeline_result,
        "archived_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        client = get_s3_client()
        ensure_bucket(client)
        client.put_object(
            Bucket=DATALAKE_BUCKET,
            Key=key,
            Body=json.dumps(record, default=str).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"Archived {channel} event to s3://{DATALAKE_BUCKET}/{key}")
    except Exception as e:
        logger.error(f"Data lake archive failed (non-fatal, request already served): {e}")
