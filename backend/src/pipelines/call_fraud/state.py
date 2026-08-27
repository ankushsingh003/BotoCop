from typing import TypedDict, Optional, List, Dict, Any


class CallFraudState(TypedDict):
    call: Dict[str, Any]  # transcript, linked_account_id/phone_number, duration_seconds, ...
    retry_feedback: Optional[str]

    features: Optional[Dict[str, Any]]
    ml_score: Optional[Dict[str, Any]]
    identity_graph: Optional[Dict[str, Any]]

    violations: List[Dict[str, Any]]
    final_status: str
    is_short_circuited: Optional[bool]
    hitl_status: Optional[str]
    case_id: Optional[str]
    stt_metadata: Optional[Dict[str, Any]]
    evidence_hash: Optional[str]
    error: List[str]




