from langgraph.graph import StateGraph, END

from backend.src.pipelines.transaction_fraud.state import TransactionFraudState
from backend.src.pipelines.transaction_fraud.nodes import (
    retrieve_rules_node,
    audit_transaction_node,
)


def create_transaction_fraud_graph():
    graph_builder = StateGraph(TransactionFraudState)

    graph_builder.add_node("retrieve_rules", retrieve_rules_node)
    graph_builder.add_node("audit_transaction", audit_transaction_node)

    graph_builder.set_entry_point("retrieve_rules")
    graph_builder.add_edge("retrieve_rules", "audit_transaction")
    graph_builder.add_edge("audit_transaction", END)

    return graph_builder.compile()


transaction_fraud_graph = create_transaction_fraud_graph()


def run_transaction_fraud_pipeline(transaction: dict, retry_feedback: str = None) -> dict:
    """Entry point the orchestrator calls. Returns the normalized
    {"violations": [...], "final_status": ...} shape the case aggregator expects."""
    result = transaction_fraud_graph.invoke({
        "transaction": transaction,
        "retry_feedback": retry_feedback,
        "violations": [],
        "error": [],
    })
    return {
        "violations": result.get("violations", []),
        "final_status": result.get("final_status", "failed"),
        "rag_sources": result.get("rag_sources", []),
    }
