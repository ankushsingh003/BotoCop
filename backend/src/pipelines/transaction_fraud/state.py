from typing import TypedDict, Optional, List, Dict, Any


class TransactionFraudState(TypedDict):
    transaction: Dict[str, Any]  # account_id, amount, currency, merchant, txn_type, timestamp, ...
    retry_feedback: Optional[str]  # set by the orchestrator if the eval agent asks for a retry

    retrieved_rules: Optional[str]
    rag_sources: Optional[List[str]]

    violations: List[Dict[str, Any]]
    final_status: str  # success | warning | failed
    error: List[str]
