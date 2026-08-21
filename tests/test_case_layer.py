import uuid

from backend.src.case.linker import link_event_to_case, resolve_entity_id
from backend.src.case.store import append_event, get_case_with_events, update_case_status
from backend.src.case.aggregator import compute_case_risk
from backend.src.case.models import CaseStatus



def test_resolve_entity_id():
    assert resolve_entity_id("transaction", {"account_id": "acct_123"}) == "acct_123"
    assert resolve_entity_id("call", {"linked_account_id": "acct_123"}) == "acct_123"
    assert resolve_entity_id("call", {"phone_number": "+15551234"}) == "+15551234"
    assert resolve_entity_id("unknown", {}) is None


def test_two_channel_events_link_to_same_case():
    entity_id = f"acct_{uuid.uuid4().hex[:8]}"

    txn_case = link_event_to_case("transaction", {"account_id": entity_id})
    append_event(
        txn_case.case_id,
        channel="transaction",
        pipeline_result={"violations": [{"severity": "medium", "description": "Unusual transfer amount"}]},
    )

    call_case = link_event_to_case("call", {"linked_account_id": entity_id})
    assert call_case.case_id == txn_case.case_id  # linked to the SAME case, not a new one

    append_event(
        call_case.case_id,
        channel="call",
        pipeline_result={"violations": [{"severity": "high", "description": "Impersonating bank support"}]},
    )

    full_case = get_case_with_events(txn_case.case_id)
    assert full_case["entity_id"] == entity_id
    assert len(full_case["events"]) == 2
    assert {e["channel"] for e in full_case["events"]} == {"transaction", "call"}


def test_risk_aggregation_escalates_only_with_two_channels():
    single_channel_events = [
        {"channel": "transaction", "pipeline_result": {"violations": [{"severity": "high"}]}},
    ]
    result_single = compute_case_risk(single_channel_events)
    assert result_single["should_escalate"] is False  # only 1 channel involved, even at high severity

    two_channel_events = [
        {"channel": "transaction", "pipeline_result": {"violations": [{"severity": "medium"}]}},
        {"channel": "call", "pipeline_result": {"violations": [{"severity": "high"}]}},
    ]
    result_two = compute_case_risk(two_channel_events)
    assert result_two["should_escalate"] is True
    assert set(result_two["channels_seen"]) == {"transaction", "call"}


def test_case_status_update():
    entity_id = f"acct_{uuid.uuid4().hex[:8]}"
    case = link_event_to_case("transaction", {"account_id": entity_id})
    update_case_status(case.case_id, CaseStatus.ESCALATED, risk_score=0.75)

    full_case = get_case_with_events(case.case_id)
    assert full_case["status"] == "escalated"
    assert full_case["risk_score"] == 0.75
