from typing import TypedDict, Optional, List, Dict, Any


class TransactionFraudState(TypedDict):
    # input
    transaction: Dict[str, Any]  # account_id, amount, currency, merchant, txn_type, timestamp, ...
    retry_feedback: Optional[str]  # set by the orchestrator if the eval agent asks for a retry

    # intermediate
    retrieved_rules: Optional[str]
    rag_sources: Optional[List[str]]

    # output (normalized to match the schema the case aggregator expects)
    violations: List[Dict[str, Any]]
    final_status: str  # success | warning | failed
    error: List[str]
