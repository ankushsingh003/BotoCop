from langgraph.graph import StateGraph , END

from backend.src.graph.state import VideoAuditState
from backend.src.graph.nodes import (
    index_video_node,
    classify_domain_node,
    auto_content_node,
    visual_compliance_node,
    merge_results_node
)


def create_graph():
    # define the graph
    graph_builder = StateGraph(VideoAuditState)

    # add nodes
    graph_builder.add_node("index_video" , index_video_node)
    graph_builder.add_node("classify_domain" , classify_domain_node)
    graph_builder.add_node("auto_content" , auto_content_node)
    graph_builder.add_node("visual_compliance" , visual_compliance_node)
    graph_builder.add_node("merge_results" , merge_results_node)

    # add edges
    graph_builder.set_entry_point("index_video")
    graph_builder.add_edge("index_video", "classify_domain")
    graph_builder.add_edge("classify_domain", "auto_content")
    graph_builder.add_edge("auto_content", "visual_compliance")
    graph_builder.add_edge("visual_compliance" , "merge_results")
    graph_builder.add_edge("merge_results" , END)

    # compile the graph
    video_audit_graph = graph_builder.compile()

    return video_audit_graph


# expose the runable video_audit_graph 
video_audit_graph = create_graph()
