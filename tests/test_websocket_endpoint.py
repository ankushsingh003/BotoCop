"""
End-to-end test of /ws/events using synthetic data.

Both the transaction (RAG + LLM) and call (LLM) pipelines require a real
GEMINI_API_KEY and network access to Google's API, neither of which is
available in this sandbox. Both pipelines are mocked so this test proves the WebSocket ->
orchestrator -> case-layer -> eval-loop wiring and the cross-channel
escalation contract are correct. In your own environment with
GEMINI_API_KEY set, swap the monkeypatched functions for the real
run_transaction_fraud_pipeline / run_call_fraud_pipeline to run this
fully unmocked.
"""
from fastapi.testclient import TestClient

from backend.src.api.server import app
from backend.src.synthetic.generator import generate_correlated_fraud_case, generate_fraud_transaction
from backend.src.orchestrator import orchestrator
from backend.src.orchestrator.eval_agent import EventEvalModel, CaseEvalModel


def _fake_transaction_pipeline(event_payload, retry_feedback=None):
    return {
        "violations": [{
            "category": "High_Risk_Corridor",
            "description": "Large wire transfer to a first-time payee in a high-risk corridor.",
            "severity": "high",
            "suggestion": "Hold for manual review before settlement.",
        }],
        "final_status": "warning",
        "rag_sources": ["rules_finance.pdf"],
    }


def _fake_call_pipeline(event_payload, retry_feedback=None):
    return {
        "violations": [{
            "category": "Bank_Impersonation",
            "description": "Transcript matches bank-impersonation scam script pattern.",
            "severity": "high",
            "suggestion": "Flag account for manual review.",
        }],
        "final_status": "warning",
        "rag_sources": [],
    }


def test_websocket_single_event_roundtrip(monkeypatch):
    monkeypatch.setitem(orchestrator.PIPELINES, "transaction", _fake_transaction_pipeline)
    monkeypatch.setattr(
        orchestrator.default_eval_agent, "evaluate_event",
        lambda *a, **kw: EventEvalModel(is_confident=True, confidence_score=0.9, feedback=""),
    )

    txn = generate_fraud_transaction()
    client = TestClient(app)
    with client.websocket_connect("/ws/events") as ws:
        ws.send_json({"channel": "transaction", "payload": txn})
        response = ws.receive_json()

    assert response["status"] == "ok"
    result = response["result"]
    assert result["event_eval"]["is_confident"] is True
    assert result["pipeline_result"]["final_status"] == "warning"


def test_websocket_cross_channel_escalation_synthetic(monkeypatch):
    call_event, txn_event = generate_correlated_fraud_case()

    monkeypatch.setitem(orchestrator.PIPELINES, "call", _fake_call_pipeline)
    monkeypatch.setitem(orchestrator.PIPELINES, "transaction", _fake_transaction_pipeline)
    monkeypatch.setattr(
        orchestrator.default_eval_agent, "evaluate_event",
        lambda *a, **kw: EventEvalModel(is_confident=True, confidence_score=0.9, feedback=""),
    )
    monkeypatch.setattr(
        orchestrator.default_eval_agent, "evaluate_case",
        lambda case: CaseEvalModel(
            is_coordinated_fraud=True,
            confidence_score=0.9,
            reasoning="Scam call immediately followed by a matching large wire transfer to a new payee.",
        ),
    )

    client = TestClient(app)
    with client.websocket_connect("/ws/events") as ws:
        ws.send_json({"channel": "call", "payload": call_event})
        r1 = ws.receive_json()

        ws.send_json({"channel": "transaction", "payload": txn_event})
        r2 = ws.receive_json()

    assert r1["status"] == "ok"
    assert r2["status"] == "ok"
    assert r2["result"]["case_id"] == r1["result"]["case_id"]  # linked to the same case
    assert r2["result"]["case_risk"]["should_escalate"] is True
    assert r2["result"]["case_eval"]["is_coordinated_fraud"] is True
