from langgraph.graph import StateGraph, END

from backend.src.pipelines.text_fraud.state import TextFraudState
from backend.src.pipelines.text_fraud.nodes import audit_text_node


def create_text_fraud_graph():
    graph_builder = StateGraph(TextFraudState)

    graph_builder.add_node("audit_text", audit_text_node)

    graph_builder.set_entry_point("audit_text")
    graph_builder.add_edge("audit_text", END)

    return graph_builder.compile()


text_fraud_graph = create_text_fraud_graph()


def run_text_fraud_pipeline(message_event: dict, retry_feedback: str = None) -> dict:
    result = text_fraud_graph.invoke({
        "message": message_event,
        "retry_feedback": retry_feedback,
        "violations": [],
        "error": [],
    })
    return {
        "violations": result.get("violations", []),
        "final_status": result.get("final_status", "failed"),
        "rag_sources": [],
    }
