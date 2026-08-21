"""
Proves metrics are actually recorded by real calls through the
orchestrator (not just that the Counter/Histogram objects exist), and
that /metrics serves valid Prometheus exposition format via the real
FastAPI app.
"""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from backend.src.api.server import app
from backend.src.orchestrator import orchestrator
from backend.src.orchestrator.eval_agent import EventEvalModel
from backend.src.monitoring.metrics import EVENTS_PROCESSED, VIOLATIONS_DETECTED


def _counter_value(counter, **labels):
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total") and all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return 0.0


def test_handle_event_increments_events_processed_metric(monkeypatch):
    def fake_pipeline(event_payload, retry_feedback=None):
        return {
            "violations": [{"category": "Test", "description": "x", "severity": "high", "suggestion": "y"}],
            "final_status": "warning",
            "rag_sources": [],
        }

    monkeypatch.setitem(orchestrator.PIPELINES, "transaction", fake_pipeline)

    mock_eval_agent = MagicMock()
    mock_eval_agent.evaluate_event.return_value = EventEvalModel(is_confident=True, confidence_score=0.9, feedback="")

    before = _counter_value(EVENTS_PROCESSED, channel="transaction", final_status="warning")
    before_violations = _counter_value(VIOLATIONS_DETECTED, channel="transaction", severity="high")

    orchestrator.handle_event(
        "transaction", {"account_id": "acct_metrics_test", "amount": 100}, eval_agent=mock_eval_agent
    )

    after = _counter_value(EVENTS_PROCESSED, channel="transaction", final_status="warning")
    after_violations = _counter_value(VIOLATIONS_DETECTED, channel="transaction", severity="high")

    assert after == before + 1
    assert after_violations == before_violations + 1


def test_metrics_endpoint_serves_prometheus_exposition_format():
    client = TestClient(app)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "botocop_events_processed_total" in body
    assert "botocop_pipeline_duration_seconds" in body


def test_eval_retries_incremented_on_non_confident_attempts(monkeypatch):
    from backend.src.monitoring.metrics import EVAL_RETRIES

    def flaky_pipeline(event_payload, retry_feedback=None):
        return {"violations": [], "final_status": "success", "rag_sources": []}

    monkeypatch.setitem(orchestrator.PIPELINES, "transaction", flaky_pipeline)
    mock_eval_agent = MagicMock()
    mock_eval_agent.evaluate_event.return_value = EventEvalModel(is_confident=False, confidence_score=0.1, feedback="retry")

    before = _counter_value(EVAL_RETRIES, channel="transaction")

    orchestrator.handle_event(
        "transaction", {"account_id": "acct_retry_test", "amount": 10}, eval_agent=mock_eval_agent
    )

    after = _counter_value(EVAL_RETRIES, channel="transaction")
    assert after == before + (orchestrator.MAX_RETRIES - 1)


def test_open_cases_gauge_computed_live_from_db():
    from backend.src.monitoring.metrics import OPEN_CASES

    value = OPEN_CASES.collect()[0].samples[0].value
    assert value >= 0
