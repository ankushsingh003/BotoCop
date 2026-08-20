"""
Real end-to-end test of the Spark batch retraining job -- actually
starts a local PySpark session and runs the real feature engineering +
retrain, against local files formatted exactly like the data lake's
archived records (same shape datalake/writer.py produces). No mocking
of Spark itself; only the storage location is local instead of s3a://,
since no real MinIO server is reachable in this sandbox.
"""
import json
import os

import pytest

from backend.src.synthetic.generator import generate_normal_transaction, generate_fraud_transaction


def _write_archive_record(dir_path, channel, transaction):
    record = {
        "channel": channel,
        "case_id": "test-case",
        "event_payload": transaction,
        "pipeline_result": {"violations": [], "final_status": "success"},
        "archived_at": "2026-08-05T00:00:00+00:00",
    }
    path = os.path.join(dir_path, f"{os.urandom(4).hex()}.json")
    with open(path, "w") as f:
        f.write(json.dumps(record, default=str))


@pytest.fixture
def local_archive(tmp_path):
    archive_dir = tmp_path / "transaction"
    archive_dir.mkdir()
    for _ in range(200):
        _write_archive_record(str(archive_dir), "transaction", generate_normal_transaction())
    return f"file://{archive_dir}"


def test_spark_retrain_reads_and_fits_on_local_archive(local_archive, tmp_path):
    from backend.spark.train_transaction_model_spark import train

    # write to an isolated model dir -- must NOT touch the real
    # backend/data/models path, which other tests depend on.
    isolated_model_dir = str(tmp_path / "models")
    model, stats = train(local_archive, contamination=0.05, model_dir=isolated_model_dir)

    assert model is not None
    assert set(stats["feature_names"]) == {"amount", "hour_of_day", "is_new_payee", "country_risk"}

    # The retrained model should still reliably flag synthetic fraud as
    # an outlier -- proves the Spark-engineered features match what the
    # live scoring path expects (features.py), not just that training ran.
    from backend.src.pipelines.transaction_fraud.features import transaction_to_features

    flagged = 0
    n = 15
    for _ in range(n):
        x = transaction_to_features(generate_fraud_transaction())
        if model.predict([x])[0] == -1:
            flagged += 1
    assert flagged / n > 0.7
