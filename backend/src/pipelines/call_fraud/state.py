from typing import TypedDict, Optional, List, Dict, Any


class CallFraudState(TypedDict):
    call: Dict[str, Any]  # transcript, linked_account_id/phone_number, duration_seconds, ...
    retry_feedback: Optional[str]

    violations: List[Dict[str, Any]]
    final_status: str
    error: List[str]
