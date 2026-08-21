import os
import sys
import base64
import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch

sys.path.append("d:\\Data_Science_3")

from backend.src.graph.nodes import visual_compliance_node, merge_results_node
from backend.src.services.video_index import select_keyframes


@patch("backend.src.graph.nodes.ChatGroq")
def test_visual_compliance_node(mock_chat_groq):
    """
    Test visual_compliance_node with a mocked Groq vision response.
    """
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm
    
    mock_response = MagicMock()
    mock_response.content = """{
        "compliance_result": [
            {
                "category": "General",
                "description": "On-screen disclaimer text is too small.",
                "severity": "Warning",
                "suggestion": "Increase the font size of the disclaimer.",
                "timestamp": "00:00:10"
            }
        ],
        "final_status": "warning",
        "final_report": "Visual audit warnings found."
    }"""
    mock_llm.invoke.return_value = mock_response
    
    state = {
        "frames": ["base64_frame_1", "base64_frame_2"],
        "video_url": "http://example.com",
        "video_id": "test_id",
    }
    
    result = visual_compliance_node(state)
    
    assert "visual_violations" in result
    assert len(result["visual_violations"]) == 1
    assert result["visual_violations"][0]["category"] == "General"
    assert result["visual_status"] == "warning"
    assert result["selected_frames"] == ["base64_frame_1", "base64_frame_2"]


def test_merge_results_node_fail_pass():
    """
    Test that audio fail + visual pass = overall fail in merge_results_node.
    """
    state = {
        "final_status": "failed",       # Audio failed
        "visual_status": "success",     # Visual passed
        "compliance_result": [{"category": "Financial", "description": "Audio issue", "severity": "Critical"}],
        "visual_violations": [{"category": "General", "description": "Visual note", "severity": "Info"}]
    }
    
    result = merge_results_node(state)
    
    assert result["final_status"] == "failed"
    assert result["compliance_result"] == [{"category": "General", "description": "Visual note", "severity": "Info"}]
    assert "Overall Audit Status: FAILED" in result["final_message"]
    assert result["merged_report"]["audio_violations_count"] == 1
    assert result["merged_report"]["visual_violations_count"] == 1
    assert result["merged_report"]["total_violations_count"] == 2


def test_select_keyframes_spacing():
    """
    Test that select_keyframes returns correct count and spacing.
    """
    frames = [f"frame_{i}" for i in range(10)]
    
    selected = select_keyframes(frames, n=5)
    assert len(selected) == 5
    assert selected == ["frame_0", "frame_2", "frame_4", "frame_6", "frame_9"]
    
    selected_3 = select_keyframes(frames, n=3)
    assert len(selected_3) == 3
    assert selected_3 == ["frame_0", "frame_4", "frame_9"]
    
    selected_all = select_keyframes(frames, n=12)
    assert selected_all == frames


def test_frame_base64_encoding_validity():
    """
    Test that base64 encoding and decoding of frames is valid and not truncated/corrupted.
    """
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    
    b64_str = base64.b64encode(buffer).decode("utf-8")
    
    assert len(b64_str) > 0
    
    decoded_bytes = base64.b64decode(b64_str)
    
    decoded_arr = np.frombuffer(decoded_bytes, dtype=np.uint8)
    decoded_img = cv2.imdecode(decoded_arr, cv2.IMREAD_COLOR)
    
    assert decoded_img is not None
    assert decoded_img.shape == (100, 100, 3)
    np.testing.assert_array_equal(decoded_img, img)


def test_api_audit_endpoint():
    """
    Test the FastAPI /api/audit endpoint with mocked LangGraph run response.
    """
    from fastapi.testclient import TestClient
    from backend.src.api.server import app

    client = TestClient(app)
    
    with patch("backend.src.graph.workflow.video_audit_graph.invoke") as mock_invoke:
        mock_invoke.return_value = {
            "video_url": "https://youtu.be/yx39ed__8ZA",
            "video_id": "test_id",
            "final_status": "success",
            "compliance_result": [
                {
                    "category": "Financial_Compliance",
                    "description": "Unregistered advisory statement detected.",
                    "severity": "Warning",
                    "suggestion": "Add registered advisor disclosure."
                },
                {
                    "category": "General",
                    "description": "Visual disclaimer logo too small.",
                    "severity": "Critical",
                    "suggestion": "Increase sizing."
                }
            ],
            "visual_violations": [
                {
                    "category": "General",
                    "description": "Visual disclaimer logo too small.",
                    "severity": "Critical",
                    "suggestion": "Increase sizing."
                }
            ],
            "rag_sources": ["guideline_1.pdf"]
        }
        
        response = client.post("/api/audit", json={"video_url": "https://youtu.be/yx39ed__8ZA"})
        assert response.status_code == 200
        data = response.json()
        
        assert data["video_url"] == "https://youtu.be/yx39ed__8ZA"
        assert data["status"] == "pass"
        assert data["domain"] == "general"
        assert len(data["audio_violations"]) == 1
        assert data["audio_violations"][0]["rule"] == "Financial_Compliance"
        assert data["audio_violations"][0]["severity"] == "medium"
        assert data["audio_violations"][0]["description"] == "Unregistered advisory statement detected."
        assert data["audio_violations"][0]["source"] == "audio"
        
        assert len(data["visual_violations"]) == 1
        assert data["visual_violations"][0]["rule"] == "General"
        assert data["visual_violations"][0]["severity"] == "high"
        assert data["visual_violations"][0]["description"] == "Visual disclaimer logo too small."
        assert data["visual_violations"][0]["source"] == "visual"
        
        assert data["rag_sources"] == ["guideline_1.pdf"]
        assert data["total_violations"] == 2

    with patch("backend.src.graph.workflow.video_audit_graph.invoke") as mock_invoke:
        mock_invoke.return_value = {
            "domain": "healthcare",
            "rag_sources": ["guideline_1.pdf"],
            "merged_report": {
                "status": "fail",
                "audio": [
                    {
                        "rule": "Healthcare_Compliance",
                        "severity": "high",
                        "description": "Unsupported medical claim.",
                        "source": "audio"
                    }
                ],
                "visual": [
                    {
                        "rule": "General",
                        "severity": "medium",
                        "description": "Logo out of bounds.",
                        "source": "visual"
                    }
                ]
            }
        }
        
        response2 = client.post("/audit", json={"url": "https://youtu.be/yx39ed__8ZA"})
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["video_url"] == "https://youtu.be/yx39ed__8ZA"
        assert data2["status"] == "fail"
        assert data2["domain"] == "healthcare"
        assert len(data2["audio_violations"]) == 1
        assert data2["audio_violations"][0]["rule"] == "Healthcare_Compliance"
        assert data2["audio_violations"][0]["severity"] == "high"
        assert len(data2["visual_violations"]) == 1
        assert data2["visual_violations"][0]["source"] == "visual"
        assert data2["total_violations"] == 2


@patch("backend.src.graph.nodes.ChatGroq")
def test_classify_domain_node(mock_chat_groq):
    """
    Test classify_domain_node with a mocked LLM classification response.
    """
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm
    
    mock_response = MagicMock()
    mock_response.content = "finance"
    mock_llm.invoke.return_value = mock_response
    
    state = {
        "transcript": "I want to buy some bitcoin and invest in crypto.",
        "video_url": "http://example.com"
    }
    
    from backend.src.graph.nodes import classify_domain_node
    result = classify_domain_node(state)
    assert result["domain"] == "finance"


@patch("backend.src.graph.nodes.ChatGroq")
@patch("backend.src.rag.retriever.RuleRetriever")
def test_auto_content_node_routing(mock_retriever_class, mock_chat_groq):
    """
    Test auto_content_node routing where the domain decides which collection is queried.
    """
    mock_llm = MagicMock()
    mock_chat_groq.return_value = mock_llm
    
    mock_retriever = MagicMock()
    mock_retriever_class.return_value = mock_retriever
    mock_retriever.retrieve.return_value = "Mocked rules content"
    
    mock_llm_response = MagicMock()
    mock_llm_response.content = """{
        "compliance_result": [],
        "final_status": "success",
        "final_report": "All clean."
    }"""
    mock_llm.invoke.return_value = mock_llm_response
    
    state = {
        "domain": "healthcare",
        "transcript": "Some medical advice transcript.",
        "video_url": "http://example.com"
    }
    
    from backend.src.graph.nodes import auto_content_node
    auto_content_node(state)
    
    mock_retriever.retrieve.assert_called_with("Some medical advice transcript.", domain="healthcare")

