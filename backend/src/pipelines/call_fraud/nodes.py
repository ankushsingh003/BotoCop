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


from backend.src.pipelines.call_fraud.blocklist import get_scam_blocklist


from backend.src.pipelines.call_fraud.stt_engine import MultilingualSTTEngine
from backend.src.pipelines.call_fraud.evidence_store import get_evidence_vault


def stt_preprocessing_node(state: CallFraudState) -> Dict[str, Any]:
    """Node -1: Process audio/transcript via Whisper STT and perform Indic translation."""
    call = state.get("call") or {}
    normalized_transcript, detected_lang, stt_meta = MultilingualSTTEngine.process_audio_or_transcript(call)

    updated_call = dict(call)
    updated_call["transcript"] = normalized_transcript

    return {
        "call": updated_call,
        "stt_metadata": stt_meta,
    }


def check_blocklist_node(state: CallFraudState) -> Dict[str, Any]:
    """Node 0: Fast O(1) Layer 4 Known-Bad Number Blocklist lookup for deterministic short-circuiting."""

    call = state.get("call") or {}
    caller_phone = call.get("caller_phone") or call.get("phone_number") or ""

    is_blocked, block_info = get_scam_blocklist().check_blocklist(caller_phone)
    if is_blocked:
        logger.warning(f"CRITICAL SHORT-CIRCUIT: Caller {caller_phone} is in known scam blocklist ({block_info['source']})")
        ml_score = {
            "fraud_probability": 1.0,
            "fraud_percentage": 100.0,
            "risk_level": "CRITICAL",
            "recommended_action": "BLOCK_CALL_AND_FREEZE_ACCOUNT",
            "top_risk_drivers": [f"Known Scam Number Blocklist (Source: {block_info['source']})"],
            "model_type": "Deterministic_Blocklist",
        }
        violations = [{
            "category": "Known_Scammer_Blocklist",
            "description": f"Caller phone matched verified cybercrime scam database ({block_info['source']}: {block_info['reason']})",
            "severity": "critical",
            "suggestion": "Block number across carrier gateway and alert user immediately"
        }]
        return {
            "is_short_circuited": True,
            "ml_score": ml_score,
            "violations": violations,
            "final_status": "failed"
        }
    return {"is_short_circuited": False}


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


from backend.src.pipelines.call_fraud.script_cache import get_script_cache


def audit_call_node(state: CallFraudState) -> Dict[str, Any]:
    """
    Node 4: Multi-stage transcript evaluation:
      Stage 4a: Low-Risk ML Pre-Filter (bypasses LLM for clear benign calls)
      Stage 4b: Scam Script Similarity Cache Lookup (bypasses LLM for near-duplicate scripts)
      Stage 4c: Gemini LLM Audit (invoked only for novel ambiguous cases)
    """
    call = state.get("call") or {}
    transcript = call.get("transcript", "")
    retry_feedback = state.get("retry_feedback")
    ml_score = state.get("ml_score") or {}
    graph = state.get("identity_graph") or {}

    # Stage 4a: Lightweight Low-Risk Pre-Filter
    high_risk_keywords = ["otp", "bank", "police", "blocked", "wire", "urgent", "warrant", "arrest", "kyc"]
    contains_trigger = any(kw in transcript.lower() for kw in high_risk_keywords)
    fraud_prob = ml_score.get("fraud_probability", 0.0)

    if fraud_prob < 0.25 and not contains_trigger:
        logger.info(f"PRE-FILTER PASSED: Low-risk call (prob={fraud_prob:.2f}), bypassing LLM audit.")
        return {"violations": [], "final_status": "success", "audit_source": "Low_Risk_PreFilter"}

    # Stage 4b: Scam Script Similarity Cache Lookup
    cache_hit, cached_violations, similarity = get_script_cache().lookup_cached_script(transcript)
    if cache_hit:
        logger.info(f"SCRIPT CACHE HIT: Reused cached audit for near-duplicate script (similarity={similarity:.2f})")
        final_status = "failed" if ml_score.get("risk_level") in ["HIGH", "CRITICAL"] else "warning"
        return {"violations": cached_violations, "final_status": final_status, "audit_source": f"Script_Cache_Hit_{similarity:.2f}"}

    # Stage 4c: Escalated Novel Case -> Gemini LLM Forensic Audit
    logger.info("ESCALATING TO GEMINI LLM: Novel ambiguous transcript requires forensic LLM reasoning pass.")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

    if not api_key:
        logger.warning("GEMINI_API_KEY not set. Using rule-based ML fallback audit.")
        violations = [{
            "category": "High_ML_Scam_Probability",
            "description": f"ML Classifier detected high fraud probability ({ml_score.get('fraud_percentage', 0)}%) based on: {', '.join(ml_score.get('top_risk_drivers', []))}",
            "severity": "high" if ml_score.get("risk_level") == "CRITICAL" else "medium",
            "suggestion": ml_score.get("recommended_action", "FLAG_SUSPICIOUS")
        }] if ml_score.get("risk_level") in ["HIGH", "CRITICAL"] else []
        final_status = "failed" if ml_score.get("risk_level") == "CRITICAL" else ("warning" if violations else "success")
        return {"violations": violations, "final_status": final_status, "audit_source": "ML_Rule_Fallback_No_API_Key"}

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
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=api_key)
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=content)])

        response_content = response.content
        start_idx = response_content.find("{")
        end_idx = response_content.rfind("}")
        json_str = response_content[start_idx:end_idx + 1] if start_idx != -1 and end_idx != -1 else response_content
        data = json.loads(json_str)
        report = CallAuditModel(**data)

        violations = [v.model_dump() for v in report.violations]

        # Cache new scam script if violations were detected
        if violations:
            get_script_cache().cache_scam_script(transcript, violations)

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

        return {"violations": violations, "final_status": final_status, "audit_source": "Gemini_LLM_Audit"}
    except Exception as e:
        logger.error(f"Call audit failed: {e}")
        return {"error": [str(e)], "violations": [], "final_status": "failed", "audit_source": "LLM_Error_Fallback"}



from backend.src.pipelines.call_fraud.hitl_queue import get_hitl_queue

def hitl_routing_node(state: CallFraudState) -> Dict[str, Any]:
    """
    Node 5: Route flagged or high-risk cases into Layer 5 Human-In-The-Loop review queue.
    """
    case_id = state.get("case_id") or str(uuid.uuid4())
    ml_score = state.get("ml_score") or {}
    violations = state.get("violations") or []
    final_status = state.get("final_status") or "success"

    # Enqueue if risk is HIGH/CRITICAL or violations exist
    needs_review = (
        ml_score.get("risk_level") in ["HIGH", "CRITICAL"]
        or len(violations) > 0
        or final_status in ["warning", "failed"]
    )

    if needs_review and not state.get("is_short_circuited"):
        queue_item = get_hitl_queue().enqueue_case(
            case_id=case_id,
            call_data=state.get("call") or {},
            ml_score=ml_score,
            violations=violations,
            identity_graph=state.get("identity_graph")
        )
        return {"hitl_status": "PENDING_HUMAN_REVIEW", "case_id": case_id}

    return {"hitl_status": "AUTO_APPROVED", "case_id": case_id}


def record_evidence_node(state: CallFraudState) -> Dict[str, Any]:
    """Node 6: Generate tamper-evident SHA-256 evidence package in Chain-of-Custody Vault."""
    case_id = state.get("case_id") or str(uuid.uuid4())
    evidence_vault = get_evidence_vault()

    record = evidence_vault.record_call_evidence(
        case_id=case_id,
        call_data=state.get("call") or {},
        features=state.get("features") or {},
        ml_score=state.get("ml_score") or {},
        violations=state.get("violations") or [],
        final_status=state.get("final_status") or "success",
        identity_graph=state.get("identity_graph"),
        stt_metadata=state.get("stt_metadata")
    )

    return {
        "case_id": case_id,
        "evidence_hash": record["hashes"]["pipeline_sha256"]
    }


