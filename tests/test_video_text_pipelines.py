from unittest.mock import patch

from backend.src.case.linker import resolve_entity_id
from backend.src.orchestrator import orchestrator


def test_resolve_entity_id_text_and_video_channels():
    assert resolve_entity_id("text", {"linked_account_id": "acct_1"}) == "acct_1"
    assert resolve_entity_id("text", {"sender_email": "a@b.com"}) == "a@b.com"
    assert resolve_entity_id("video", {"advertiser_id": "adv_1"}) == "adv_1"


def test_orchestrator_registers_all_four_channels():
    assert set(orchestrator.PIPELINES.keys()) == {"transaction", "call", "text", "video"}
    assert all(orchestrator.PIPELINE_REQUIRES_EVAL.values())  # all LLM-based, eval loop applies to all


def test_video_pipeline_wrapper_normalizes_merged_report():
    from backend.src.pipelines.video_compliance.workflow import run_video_compliance_pipeline

    fake_graph_output = {
        "final_status": "warning",
        "rag_sources": ["youtube-ad-specs.pdf"],
        "merged_report": {
            "audio": [{"rule": "Undisclosed_Sponsorship", "severity": "medium", "description": "No #ad tag found", "source": "audio"}],
            "visual": [{"rule": "Prohibited_Claim", "severity": "high", "description": "On-screen text makes unverified claim", "source": "visual"}],
        },
    }

    with patch("backend.src.pipelines.video_compliance.workflow.video_audit_graph") as mock_graph:
        mock_graph.invoke.return_value = fake_graph_output
        result = run_video_compliance_pipeline({"video_url": "https://youtube.com/watch?v=abc", "video_id": "abc"})

    assert result["final_status"] == "warning"
    assert len(result["violations"]) == 2
    categories = {v["category"] for v in result["violations"]}
    assert categories == {"Undisclosed_Sponsorship", "Prohibited_Claim"}
