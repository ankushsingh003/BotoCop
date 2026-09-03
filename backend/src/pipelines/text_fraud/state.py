from typing import TypedDict, Optional, List, Dict, Any


class TextFraudState(TypedDict):
    message: Dict[str, Any]  # body, subject, sender_email, auth_results_header, linked_account_id, ...
    retry_feedback: Optional[str]

    violations: List[Dict[str, Any]]
    final_status: str
    error: List[str]
