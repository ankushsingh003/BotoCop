from typing import TypedDict, Optional, List, Dict, Any


class TextFraudState(TypedDict):
    message: Dict[str, Any]  # body, subject, sender, linked_account_id/sender_email, ...
    retry_feedback: Optional[str]

    violations: List[Dict[str, Any]]
    final_status: str
    error: List[str]
