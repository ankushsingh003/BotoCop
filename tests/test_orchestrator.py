import uuid

from unittest.mock import MagicMock

from backend.src.case.store import get_case_with_events
from backend.src.orchestrator import orchestrator
from backend.src.orchestrator.eval_agent import EventEvalModel, CaseEvalModel

# DB setup/teardown is handled once for the whole session by tests/conftest.py


def fake_transaction_pipeline(event_payload, retry_feedback=None):
    return {
        "violations": [{
            "category": "Unusual_Velocity",
            "description": "Large transfer to a first-time payee",
            "severity": "medium",
            "suggestion": "Hold for manual review",
        }],
        "final_status": "warning",
        "rag_sources": ["rules_finance.pdf"],
    }


def fake_call_pipeline(event_payload, retry_feedback=None):
    transcript = (event_payload.get("transcript") or "").lower()
    violations = []
    if any(kw in transcript for kw in ["wire transfer", "gift card", "bank support", "verify your account"]):
        violations.append({
            "category": "Impersonation",
            "description": "Transcript contains phrasing consistent with bank-impersonation scam scripts.",
            "severity": "high",
            "suggestion": "Flag account for manual review before any linked transaction is approved.",
        })
    return {"violations": violations, "final_status": "warning" if violations else "success", "rag_sources": []}


def test_orchestrator_links_two_channels_and_escalates(monkeypatch):
    entity_id = f"acct_{uuid.uuid4().hex[:8]}"
    monkeypatch.setitem(orchestrator.PIPELINES, "transaction", fake_transaction_pipeline)
    monkeypatch.setitem(orchestrator.PIPELINES, "call", fake_call_pipeline)

    mock_eval_agent = MagicMock()
    mock_eval_agent.evaluate_event.return_value = EventEvalModel(
        is_confident=True, confidence_score=0.9, feedback=""
    )
    mock_eval_agent.evaluate_case.return_value = CaseEvalModel(
        is_coordinated_fraud=True,
        confidence_score=0.85,
        reasoning="Scam-call script matched a same-day large transfer to a new payee.",
    )

    result1 = orchestrator.handle_event(
        "transaction",
        {"account_id": entity_id, "amount": 5000, "currency": "USD",
         "merchant": "Unknown Payee", "is_new_payee": True},
        eval_agent=mock_eval_agent,
    )
    assert result1["case_risk"]["should_escalate"] is False  # only 1 channel so far

    # call channel mocked too (both pipelines are LLM-based now, no network here)
    result2 = orchestrator.handle_event(
        "call",
        {"linked_account_id": entity_id,
         "transcript": "This is your bank support, please confirm a wire transfer."},
        eval_agent=mock_eval_agent,
    )
    assert result2["case_id"] == result1["case_id"]  # linked to the SAME case
    assert result2["case_risk"]["should_escalate"] is True
    assert result2["case_eval"]["is_coordinated_fraud"] is True

    full_case = get_case_with_events(result2["case_id"])
    assert full_case["status"] == "escalated"
    assert len(full_case["events"]) == 2


def test_orchestrator_bounded_retry_loop(monkeypatch):
    entity_id = f"acct_{uuid.uuid4().hex[:8]}"
    call_count = {"n": 0}

    def flaky_pipeline(event_payload, retry_feedback=None):
        call_count["n"] += 1
        return {"violations": [], "final_status": "success", "rag_sources": []}

    monkeypatch.setitem(orchestrator.PIPELINES, "transaction", flaky_pipeline)

    mock_eval_agent = MagicMock()
    mock_eval_agent.evaluate_event.return_value = EventEvalModel(
        is_confident=False, confidence_score=0.2, feedback="Not grounded, retry"
    )
    mock_eval_agent.evaluate_case.return_value = CaseEvalModel(
        is_coordinated_fraud=False, confidence_score=0.1, reasoning="n/a"
    )

    orchestrator.handle_event(
        "transaction",
        {"account_id": entity_id, "amount": 10},
        eval_agent=mock_eval_agent,
    )

    # never-confident eval must not loop forever -- stops at the bound
    assert call_count["n"] == orchestrator.MAX_RETRIES
