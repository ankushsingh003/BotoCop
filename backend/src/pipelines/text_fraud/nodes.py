import json
import logging
import os
import uuid
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.src.pipelines.text_fraud.state import TextFraudState
from backend.src.pipelines.text_fraud.auth_headers import (
    parse_authentication_results,
    get_deterministic_severity_floor,
    format_auth_context_block,
)

logger = logging.getLogger("text-fraud")


class TextViolationModel(BaseModel):
    category: str = Field(description="e.g. Phishing_Link, Credential_Harvest, Spoofed_Sender, Urgency_Pressure")
    description: str
    severity: str = Field(description="low, medium, or high")
    suggestion: str


class TextAuditModel(BaseModel):
    violations: List[TextViolationModel] = Field(default_factory=list)
    final_status: str = Field(description="success, warning, or failed")


_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


def _apply_auth_floor(violations: List[dict], final_status: str, auth, floor: str) -> tuple:
    """
    Applies the deterministic DMARC-fail floor from auth_headers.py.
    Adds an explicit, cited violation for the auth failure itself
    (rather than silently inflating an existing violation's severity),
    so the reason a case is at least "medium" is visible and explainable
    in the stored result -- consistent with how every other deterministic
    signal in this project (e.g. the call pipeline's blocklist) is
    surfaced as its own citable finding, not folded invisibly into a
    number.
    """
    already_has_auth_violation = any(v.get("category") == "Sender_Authentication_Failure" for v in violations)
    if not already_has_auth_violation:
        violations = violations + [{
            "category": "Sender_Authentication_Failure",
            "description": (
                f"This message failed DMARC authentication (DMARC={auth['dmarc_result']}, "
                f"SPF={auth['spf_result']}, DKIM={auth['dkim_result']}) -- the sending domain "
                f"did not pass the receiving mail server's authenticity check, independent of "
                f"what the message content says."
            ),
            "severity": floor,
            "suggestion": "Verify sender domain identity before acting on this message's requests.",
        }]

    # Ensure final_status reflects the floor even if every LLM-judged
    # violation individually scored below it.
    highest_severity = max(
        (_SEVERITY_RANK.get(v.get("severity", "low"), 0) for v in violations),
        default=0,
    )
    if _SEVERITY_RANK.get(floor, 0) >= highest_severity and final_status == "success":
        final_status = "warning"

    return violations, final_status


def audit_text_node(state: TextFraudState) -> Dict[str, Any]:
    """
    Direct LLM classification of message/email content for phishing and
    scam indicators, layered with a deterministic SPF/DKIM/DMARC check
    (see auth_headers.py) that doesn't depend on the LLM correctly
    weighing authentication on its own. No RAG here for the same reason
    as the call pipeline -- this is a text-understanding task the LLM
    can do directly, not a lookup against an external policy document.
    """
    message = state.get("message") or {}
    body = message.get("body", "")
    subject = message.get("subject", "")
    # NOTE: this key must match what case/linker.py and the golden
    # eval dataset use for entity resolution ("sender_email"), not a
    # differently-named field -- this was previously reading "sender",
    # a key nothing else in the system ever populated, so the LLM never
    # actually saw the sender's address in any audit.
    sender = message.get("sender_email", "")
    retry_feedback = state.get("retry_feedback")

    # Authentication-Results is the raw header string, if the caller has
    # it (e.g. a real inbound-parse webhook adapter reading it straight
    # off the received email). Absence is expected and handled -- most
    # test/synthetic payloads won't have this yet.
    auth = parse_authentication_results(message.get("auth_results_header"))
    auth_floor = get_deterministic_severity_floor(auth)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash")

    if not api_key:
        logger.warning("GEMINI_API_KEY not set. Text fraud audit did NOT run.")
        # Fail CLOSED, not open: a missing key must not look identical to
        # "checked and found clean." A case store/reviewer needs to be
        # able to tell "never actually audited" apart from "audited, no
        # violations found" -- returning final_status="success" here
        # would silently approve every message during a misconfiguration
        # or outage, which is the wrong default for a fraud detector.
        violations = []
        final_status = "needs_review"
        if auth_floor:
            violations, final_status = _apply_auth_floor(violations, "needs_review", auth, auth_floor)
        return {
            "violations": violations,
            "final_status": final_status,
            "error": ["GEMINI_API_KEY not configured; content analysis was skipped"],
        }

    cache_buster = str(uuid.uuid4())
    system_prompt = (
        f"Session ID: {cache_buster}. You are a Fraud & Phishing Analyst "
        f"specializing in email/SMS scam detection."
    )
    feedback_block = (
        f"\n<prior_attempt_feedback>\n{retry_feedback}\n</prior_attempt_feedback>\n"
        if retry_feedback else ""
    )
    auth_block = f"\n<sender_authentication_check>\n{format_auth_context_block(auth)}\n</sender_authentication_check>\n"

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
{auth_block}{feedback_block}
The sender_authentication_check block above was computed by the receiving
mail server (SPF/DKIM/DMARC), independent of message content -- treat a
DMARC=fail result as strong evidence of sender spoofing even if the body
text itself reads as plausible.

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
        final_status = report.final_status

        if auth_floor:
            violations, final_status = _apply_auth_floor(violations, final_status, auth, auth_floor)

        return {"violations": violations, "final_status": final_status}
    except Exception as e:
        logger.error(f"Text audit failed: {e}")
        violations: List[dict] = []
        final_status = "failed"
        if auth_floor:
            violations, final_status = _apply_auth_floor(violations, final_status, auth, auth_floor)
        return {"error": [str(e)], "violations": violations, "final_status": final_status}
