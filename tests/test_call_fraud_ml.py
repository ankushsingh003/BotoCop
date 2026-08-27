import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch, MagicMock
import pytest


from backend.src.pipelines.call_fraud.ml_features import extract_call_features
from backend.src.pipelines.call_fraud.ml_model import get_call_ml_model
from backend.src.pipelines.call_fraud.identity_graph import resolve_call_identity_graph
from backend.src.pipelines.call_fraud.workflow import run_call_fraud_pipeline


def test_call_ml_feature_extraction():
    sample_scam_event = {
        "caller_phone": "+919876543210",
        "linked_account_id": "ACCT-9901",
        "transcript": "Hello, I am calling from HDFC Bank. Your account will be blocked immediately! Share your OTP code to unblock.",
        "duration_seconds": 120,
        "complaint_history_count": 3,
        "hour_of_day": 22
    }

    features = extract_call_features(sample_scam_event)
    assert features.otp_request_detected == 1
    assert features.urgency_score > 0.0
    assert features.impersonation_score > 0.0
    assert features.complaint_history_count == 3
    assert features.off_hours_call == 1


def test_call_ml_model_prediction():
    model = get_call_ml_model()
    sample_scam_event = {
        "caller_phone": "+919876543210",
        "transcript": "URGENT: State Bank security alert. Account blocked! Give OTP now or police will come.",
        "duration_seconds": 45,
        "prior_complaints": 4,
        "hour_of_day": 23
    }
    features = extract_call_features(sample_scam_event)
    prediction = model.predict(features)

    assert "fraud_probability" in prediction
    assert prediction["fraud_probability"] > 0.5
    assert prediction["risk_level"] in ["HIGH", "CRITICAL"]
    assert "BLOCK" in prediction["recommended_action"] or "ALERT" in prediction["recommended_action"]
    assert len(prediction["top_risk_drivers"]) > 0


def test_call_identity_graph_resolution():
    sample_event = {
        "caller_phone": "+919812345678",
        "linked_account_id": "ACCT-8877",
        "device_id": "IMEI-8827161",
        "ip_address": "103.45.12.8",
        "prior_complaints": 2
    }

    graph = resolve_call_identity_graph(sample_event)
    assert graph["caller_phone"] == "+919812345678"
    assert graph["target_account_id"] == "ACCT-8877"
    assert len(graph["linked_cases"]) == 2
    assert any(n["type"] == "DEVICE" for n in graph["graph_nodes"])
    assert any(e["relation"] == "MENTIONED_IN" for e in graph["graph_edges"])


def test_end_to_end_call_fraud_pipeline():
    sample_event = {
        "caller_phone": "+919988776655",
        "linked_account_id": "ACCT-5544",
        "transcript": "Urgent alert: Account suspended. Tell me your bank OTP immediately.",
        "duration_seconds": 60,
        "complaint_history_count": 2,
    }

    fake_llm_response = MagicMock()
    fake_llm_response.content = '{"violations": [{"category": "Bank_Impersonation", "description": "Demanded OTP claiming bank suspension", "severity": "high", "suggestion": "Block phone and alert customer"}], "final_status": "warning"}'

    with patch("backend.src.pipelines.call_fraud.nodes.ChatGoogleGenerativeAI") as mock_llm_cls:
        mock_llm_inst = MagicMock()
        mock_llm_inst.invoke.return_value = fake_llm_response
        mock_llm_cls.return_value = mock_llm_inst

        result = run_call_fraud_pipeline(sample_event)

        assert "violations" in result
        assert "ml_score" in result
        assert "identity_graph" in result
        assert result["ml_score"]["fraud_probability"] > 0.0
        assert result["identity_graph"]["caller_phone"] == "+919988776655"
        assert len(result["violations"]) >= 1


if __name__ == "__main__":
    print("Testing ML Feature Extraction...")
    test_call_ml_feature_extraction()
    print("Testing ML Classifier Model...")
    test_call_ml_model_prediction()
    print("Testing Identity Graph Resolution...")
    test_call_identity_graph_resolution()
    print("Testing End-to-End Call Fraud Pipeline...")
    test_end_to_end_call_fraud_pipeline()
    print("ALL ML CALL FRAUD TESTS PASSED SUCCESSFULLY!")

