from langgraph.graph import StateGraph, END

from backend.src.pipelines.call_fraud.state import CallFraudState
from backend.src.pipelines.call_fraud.nodes import audit_call_node


def create_call_fraud_graph():
    graph_builder = StateGraph(CallFraudState)

    graph_builder.add_node("audit_call", audit_call_node)

    graph_builder.set_entry_point("audit_call")
    graph_builder.add_edge("audit_call", END)

    return graph_builder.compile()


call_fraud_graph = create_call_fraud_graph()


def run_call_fraud_pipeline(call_event: dict, retry_feedback: str = None) -> dict:
    result = call_fraud_graph.invoke({
        "call": call_event,
        "retry_feedback": retry_feedback,
        "violations": [],
        "error": [],
    })
    return {
        "violations": result.get("violations", []),
        "final_status": result.get("final_status", "failed"),
        "rag_sources": [],
    }
