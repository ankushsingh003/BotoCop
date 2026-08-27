import json
import logging
import os
import uuid
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.src.pipelines.call_fraud.state import CallFraudState

logger = logging.getLogger("call-fraud")


class CallViolationModel(BaseModel):
    category: str = Field(description="e.g. Bank_Impersonation, Urgency_Pressure, Gift_Card_Request, OTP_Request")
    description: str
    severity: str = Field(description="low, medium, or high")
    suggestion: str


class CallAuditModel(BaseModel):
    violations: List[CallViolationModel] = Field(default_factory=list)
    final_status: str = Field(description="success, warning, or failed")


def audit_call_node(state: CallFraudState) -> Dict[str, Any]:
    """
    Direct LLM classification of the transcript for scam-call patterns.
    No RAG here -- unlike compliance-vs-rulebook checking, "does this
    sound like a bank-impersonation or urgency-pressure scam script" is a
    task an LLM can do directly from the transcript, without needing an
    external reference document.
    """
    call = state.get("call") or {}
    transcript = call.get("transcript", "")
    retry_feedback = state.get("retry_feedback")

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=api_key)

    cache_buster = str(uuid.uuid4())
    system_prompt = (
        f"Session ID: {cache_buster}. You are a Fraud Call Analyst specializing "
        f"in scam-call and voice-phishing detection."
    )
    feedback_block = (
        f"\n<prior_attempt_feedback>\n{retry_feedback}\n</prior_attempt_feedback>\n"
        if retry_feedback else ""
    )

    content = f"""Request ID: {cache_buster}
Analyze this call transcript for scam/fraud indicators: bank impersonation,
urgency/fear pressure tactics, requests for OTP codes, gift cards, or wire
transfers, claims of being a government agency demanding payment, etc.

<transcript>
{transcript}
</transcript>
{feedback_block}
Output ONLY a valid JSON object, no preamble or markdown:
{{
    "violations": [
        {{
            "category": "Bank_Impersonation/Urgency_Pressure/Gift_Card_Request/OTP_Request/General",
            "description": "Specific finding",
            "severity": "low/medium/high",
            "suggestion": "Recommended action"
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
        return {"violations": violations, "final_status": report.final_status}
    except Exception as e:
        logger.error(f"Call audit failed: {e}")
        return {"error": [str(e)], "violations": [], "final_status": "failed"}
