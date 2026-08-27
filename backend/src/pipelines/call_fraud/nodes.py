import json
import logging
import os
import uuid
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.src.pipelines.call_fraud.state import CallFraudState
from backend.src.pipelines.call_fraud.ml_features import extract_call_features
from backend.src.pipelines.call_fraud.ml_model import get_call_ml_model
from backend.src.pipelines.call_fraud.identity_graph import resolve_call_identity_graph

logger = logging.getLogger("call-fraud")


class CallViolationModel(BaseModel):
    category: str = Field(description="e.g. Bank_Impersonation, Urgency_Pressure, Gift_Card_Request, OTP_Request, Telecom_Spoofing")
    description: str
    severity: str = Field(description="low, medium, or high")
    suggestion: str


class CallAuditModel(BaseModel):
    violations: List[CallViolationModel] = Field(default_factory=list)
    final_status: str = Field(description="success, warning, or failed")


def extract_features_node(state: CallFraudState) -> Dict[str, Any]:
    """Node 1: Extract automated ML feature vector from incoming call metadata & transcript."""
    call = state.get("call") or {}
    logger.info("Extracting ML call features...")
    features = extract_call_features(call)
    return {"features": features.model_dump()}


def ml_risk_scoring_node(state: CallFraudState) -> Dict[str, Any]:
    """Node 2: Evaluate Random Forest ML Fraud Classifier on extracted call features."""
    features_dict = state.get("features") or {}
    logger.info(f"Running ML Call Fraud Classifier on features: {features_dict}")
    from backend.src.pipelines.call_fraud.ml_features import CallFeatures
    features_obj = CallFeatures(**features_dict)
    
    ml_model = get_call_ml_model()
    ml_result = ml_model.predict(features_obj)
    logger.info(f"ML Call Fraud Prediction: Prob={ml_result['fraud_probability']}, Risk={ml_result['risk_level']}")
    return {"ml_score": ml_result}


def identity_correlation_node(state: CallFraudState) -> Dict[str, Any]:
    """Node 3: Correlate call identifiers with internal account_id, caller phone, IMEI, & prior complaints."""
    call = state.get("call") or {}
    logger.info("Resolving identity graph and cross-case complaint history...")
    graph = resolve_call_identity_graph(call)
    return {"identity_graph": graph}


def audit_call_node(state: CallFraudState) -> Dict[str, Any]:
    """
    Node 4: Gemini LLM classification of transcript enriched with ML prediction,
    explainable feature drivers, and identity graph context.
    """
    call = state.get("call") or {}
    transcript = call.get("transcript", "")
    retry_feedback = state.get("retry_feedback")
    ml_score = state.get("ml_score") or {}
    graph = state.get("identity_graph") or {}

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=api_key)

    cache_buster = str(uuid.uuid4())
    system_prompt = (
        f"Session ID: {cache_buster}. You are an AI Fraud Call & Vishing Analyst specializing "
        f"in automated voice-phishing detection, telecom fraud correlation, and risk audit."
    )
    feedback_block = (
        f"\n<prior_attempt_feedback>\n{retry_feedback}\n</prior_attempt_feedback>\n"
        if retry_feedback else ""
    )

    ml_context = json.dumps(ml_score, indent=2)
    graph_context = json.dumps({
        "caller_phone": graph.get("caller_phone"),
        "target_account_id": graph.get("target_account_id"),
        "prior_complaint_count": graph.get("prior_complaint_count"),
        "shared_device_flag": graph.get("shared_device_flag"),
        "linked_cases": graph.get("linked_cases")
    }, indent=2)

    content = f"""Request ID: {cache_buster}
Analyze this call transcript for scam/fraud indicators (bank impersonation, urgency/fear pressure tactics,
requests for OTP codes, gift cards, wire transfers, government impersonation) combined with ML model signals.

<ml_model_assessment>
{ml_context}
</ml_model_assessment>

<identity_and_cross_case_graph>
{graph_context}
</identity_and_cross_case_graph>

<transcript>
{transcript}
</transcript>
{feedback_block}
Output ONLY a valid JSON object, no preamble or markdown:
{{
    "violations": [
        {{
            "category": "Bank_Impersonation/Urgency_Pressure/Gift_Card_Request/OTP_Request/Telecom_Spoofing/General",
            "description": "Specific finding supported by transcript and ML/graph context",
            "severity": "low/medium/high",
            "suggestion": "Recommended action (e.g. block caller, notify I4C/bank, alert victim)"
        }}
    ],
    "final_status": "success/warning/failed"
}}"""

    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=content)])
        response_content = response.content
        start_idx = response_content.find("{")
        end_idx = response_content.rfind("}")
        json_str = response_content[start_idx:end_idx + 1] if start_idx != -1 and end_idx != -1 else response_content
        data = json.loads(json_str)
        report = CallAuditModel(**data)

        violations = [v.model_dump() for v in report.violations]
        
        # If ML score indicates high/critical risk and violations list is empty, synthesize ML rule violation
        if ml_score.get("risk_level") in ["HIGH", "CRITICAL"] and not violations:
            violations.append({
                "category": "High_ML_Scam_Probability",
                "description": f"ML Model detected high fraud probability ({ml_score.get('fraud_percentage', 0)}%) based on: {', '.join(ml_score.get('top_risk_drivers', []))}",
                "severity": "high" if ml_score.get("risk_level") == "CRITICAL" else "medium",
                "suggestion": ml_score.get("recommended_action", "FLAG_SUSPICIOUS")
            })

        # Map final status: high/critical risk -> warning/failed
        final_status = report.final_status
        if ml_score.get("risk_level") == "CRITICAL":
            final_status = "failed"
        elif ml_score.get("risk_level") == "HIGH" and final_status == "success":
            final_status = "warning"

        return {"violations": violations, "final_status": final_status}
    except Exception as e:
        logger.error(f"Call audit failed: {e}")
        return {"error": [str(e)], "violations": [], "final_status": "failed"}
