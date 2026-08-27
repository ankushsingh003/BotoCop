from langgraph.graph import StateGraph, END

from backend.src.pipelines.call_fraud.state import CallFraudState
from backend.src.pipelines.call_fraud.nodes import (
    check_blocklist_node,
    extract_features_node,
    ml_risk_scoring_node,
    identity_correlation_node,
    audit_call_node,
    hitl_routing_node,
)


def should_short_circuit(state: CallFraudState) -> str:
    """Route directly to END if caller is in known scam blocklist."""
    if state.get("is_short_circuited"):
        return "short_circuit"
    return "continue"


def create_call_fraud_graph():
    graph_builder = StateGraph(CallFraudState)

    graph_builder.add_node("check_blocklist", check_blocklist_node)
    graph_builder.add_node("extract_features", extract_features_node)
    graph_builder.add_node("ml_risk_scoring", ml_risk_scoring_node)
    graph_builder.add_node("identity_correlation", identity_correlation_node)
    graph_builder.add_node("audit_call", audit_call_node)
    graph_builder.add_node("hitl_routing", hitl_routing_node)

    graph_builder.set_entry_point("check_blocklist")
    graph_builder.add_conditional_edges(
        "check_blocklist",
        should_short_circuit,
        {
            "short_circuit": END,
            "continue": "extract_features",
        }
    )
    graph_builder.add_edge("extract_features", "ml_risk_scoring")
    graph_builder.add_edge("ml_risk_scoring", "identity_correlation")
    graph_builder.add_edge("identity_correlation", "audit_call")
    graph_builder.add_edge("audit_call", "hitl_routing")
    graph_builder.add_edge("hitl_routing", END)

    return graph_builder.compile()



call_fraud_graph = create_call_fraud_graph()


def run_call_fraud_pipeline(call_event: dict, retry_feedback: str = None) -> dict:
    """
    Automated Machine Learning & LLM Call Fraud Detection Pipeline.
    Integrates feature extraction, sklearn classifier scoring, cross-case graph correlation,
    forensic LLM audit, and Layer 5 Human-In-The-Loop review routing.
    """
    result = call_fraud_graph.invoke({
        "call": call_event,
        "retry_feedback": retry_feedback,
        "features": None,
        "ml_score": None,
        "identity_graph": None,
        "violations": [],
        "error": [],
    })

    return {
        "case_id": result.get("case_id"),
        "violations": result.get("violations", []),
        "final_status": result.get("final_status", "failed"),
        "hitl_status": result.get("hitl_status", "AUTO_APPROVED"),
        "ml_score": result.get("ml_score", {}),
        "identity_graph": result.get("identity_graph", {}),
        "features": result.get("features", {}),
        "rag_sources": [],
    }

