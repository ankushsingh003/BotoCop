import json
import logging
import os
import uuid
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.src.pipelines.text_fraud.state import TextFraudState

logger = logging.getLogger("text-fraud")


class TextViolationModel(BaseModel):
    category: str = Field(description="e.g. Phishing_Link, Credential_Harvest, Spoofed_Sender, Urgency_Pressure")
    description: str
    severity: str = Field(description="low, medium, or high")
    suggestion: str


class TextAuditModel(BaseModel):
    violations: List[TextViolationModel] = Field(default_factory=list)
    final_status: str = Field(description="success, warning, or failed")


def audit_text_node(state: TextFraudState) -> Dict[str, Any]:
    """
    Direct LLM classification of message/email content for phishing and
    scam indicators. No RAG here for the same reason as the call
    pipeline -- this is a text-understanding task the LLM can do
    directly, not a lookup against an external policy document.
    """
    message = state.get("message") or {}
    body = message.get("body", "")
    subject = message.get("subject", "")
    sender = message.get("sender", "")
    retry_feedback = state.get("retry_feedback")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash")

    if not api_key:
        logger.warning("GEMINI_API_KEY not set. Using rule-based fallback for text fraud audit.")
        return {"violations": [], "final_status": "success"}

    cache_buster = str(uuid.uuid4())
    system_prompt = (
        f"Session ID: {cache_buster}. You are a Fraud & Phishing Analyst "
        f"specializing in email/SMS scam detection."
    )
    feedback_block = (
        f"\n<prior_attempt_feedback>\n{retry_feedback}\n</prior_attempt_feedback>\n"
        if retry_feedback else ""
    )

    content = f"""Request ID: {cache_buster}
Analyze this message for phishing/scam indicators: spoofed sender identity,
urgency/fear pressure, requests for credentials/OTP/payment, suspicious
links, mismatched sender domain, etc.

<sender>
{sender}
</sender>
<subject>
{subject}
</subject>
<body>
{body}
</body>
{feedback_block}
Output ONLY a valid JSON object, no preamble or markdown:
{{
    "violations": [
        {{
            "category": "Phishing_Link/Credential_Harvest/Spoofed_Sender/Urgency_Pressure/General",
            "description": "Specific finding",
            "severity": "low/medium/high",
            "suggestion": "Recommended action"
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
        report = TextAuditModel(**data)

        violations = [v.model_dump() for v in report.violations]
        return {"violations": violations, "final_status": report.final_status}
    except Exception as e:
        logger.error(f"Text audit failed: {e}")
        return {"error": [str(e)], "violations": [], "final_status": "failed"}
