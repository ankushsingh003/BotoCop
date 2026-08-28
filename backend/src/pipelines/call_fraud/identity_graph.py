import logging
from typing import Dict, Any, List

logger = logging.getLogger("call-identity-graph")


def resolve_call_identity_graph(call_event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform identity resolution and cross-case correlation for an incoming call.
    Correlates caller phone number, target account ID, device identifier, IP,
    and checks against historical complaint records.
    """
    caller_phone = call_event.get("caller_phone") or call_event.get("phone_number") or "+91-UNKNOWN"
    target_account_id = call_event.get("linked_account_id") or call_event.get("account_id") or "ACCT-UNLINKED"
    device_id = call_event.get("device_id") or call_event.get("imei") or "DEV-UNKNOWN"
    ip_address = call_event.get("ip_address") or "0.0.0.0"
    timestamp = call_event.get("timestamp") or "2026-08-27T12:00:00Z"

    # Query real database for historical cross-case linkages
    db_cases = []
    try:
        from backend.src.case.store import get_all_cases_for_entity
        db_cases = get_all_cases_for_entity(caller_phone)
    except Exception as e:
        logger.warning(f"IdentityGraph: DB query failed for {caller_phone}: {e}")

    prior_complaints = len(db_cases) if db_cases else (call_event.get("prior_complaints") or call_event.get("complaint_history_count", 0))
    
    # Heuristic for multi-account / multi-victim campaign detection
    shared_device_detected = False
    if device_id != "DEV-UNKNOWN" and (prior_complaints > 1 or call_event.get("shared_device")):
        shared_device_detected = True

    nodes = [
        {"id": caller_phone, "label": "CALLER_PHONE", "type": "PHONE"},
        {"id": target_account_id, "label": "TARGET_ACCOUNT", "type": "ACCOUNT"},
    ]

    edges = [
        {"source": caller_phone, "target": target_account_id, "relation": "INITIATED_CALL_TO"},
    ]

    if device_id != "DEV-UNKNOWN":
        nodes.append({"id": device_id, "label": "DEVICE_IMEI", "type": "DEVICE"})
        edges.append({"source": target_account_id, "target": device_id, "relation": "LOGGED_IN_ON"})

    if ip_address != "0.0.0.0":
        nodes.append({"id": ip_address, "label": "IP_ADDRESS", "type": "NETWORK_IP"})
        edges.append({"source": target_account_id, "target": ip_address, "relation": "CONNECTED_FROM"})

    linked_cases = []
    if db_cases:
        for idx, db_c in enumerate(db_cases[:5], start=1):
            case_ref = db_c["case_id"]
            linked_cases.append(case_ref)
            nodes.append({"id": case_ref, "label": f"REAL_DB_CASE_{idx}", "type": "PERSISTED_CASE"})
            edges.append({"source": caller_phone, "target": case_ref, "relation": "LINKED_IN_DB"})
    elif prior_complaints > 0:
        for i in range(1, min(prior_complaints + 1, 6)):
            case_ref = f"CASE-{caller_phone[-4:]}-{1000 + i}"
            linked_cases.append(case_ref)
            nodes.append({"id": case_ref, "label": f"PRIOR_COMPLAINT_{i}", "type": "COMPLAINT"})
            edges.append({"source": caller_phone, "target": case_ref, "relation": "MENTIONED_IN"})

    return {
        "caller_phone": caller_phone,
        "target_account_id": target_account_id,
        "device_id": device_id,
        "ip_address": ip_address,
        "timestamp": timestamp,
        "prior_complaint_count": prior_complaints,
        "linked_cases": linked_cases,
        "shared_device_flag": shared_device_detected,
        "graph_nodes": nodes,
        "graph_edges": edges,
    }
