"""
Tests the data lake writer against a mocked S3 (moto), since no real
MinIO server is reachable in this sandbox. boto3 talks to MinIO and AWS
S3 identically (same API), so this proves the write/partitioning logic
correctly; pointing MINIO_ENDPOINT at a real MinIO container is the only
thing that changes to run this against the real thing.
"""
import json

from moto import mock_aws
import boto3

from backend.src.datalake.writer import archive_event
from backend.src.datalake.config import DATALAKE_BUCKET


@mock_aws
def test_archive_event_writes_partitioned_json(monkeypatch):
    import backend.src.datalake.client as client_module
    monkeypatch.setattr(client_module, "MINIO_ENDPOINT", None)

    archive_event(
        channel="transaction",
        event_payload={"account_id": "acct_1", "amount": 500},
        pipeline_result={"violations": [], "final_status": "success"},
        case_id="case-123",
    )

    client = boto3.client("s3", region_name="us-east-1")
    objects = client.list_objects_v2(Bucket=DATALAKE_BUCKET)
    keys = [o["Key"] for o in objects.get("Contents", [])]

    assert len(keys) == 1
    assert keys[0].startswith("transaction/dt=")
    assert keys[0].endswith(".json")

    body = client.get_object(Bucket=DATALAKE_BUCKET, Key=keys[0])["Body"].read()
    record = json.loads(body)
    assert record["channel"] == "transaction"
    assert record["case_id"] == "case-123"
    assert record["event_payload"]["account_id"] == "acct_1"


@mock_aws
def test_archive_event_never_raises_on_failure(monkeypatch):
    import backend.src.datalake.writer as writer_module

    def broken_client():
        raise ConnectionError("MinIO unreachable")

    monkeypatch.setattr(writer_module, "get_s3_client", broken_client)

    archive_event("call", {"linked_account_id": "x"}, {"violations": []}, case_id="case-456")
