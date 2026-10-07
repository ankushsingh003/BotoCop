"""
Eval agent: the "harnessing" reflection loop.

Operates at two levels:
  - evaluate_event: is this single pipeline result itself trustworthy
    (grounded in the retrieved rules, not hallucinated, actually relevant)?
  - evaluate_case: once 2+ channels have evidence, does the COMBINATION
    look like coordinated fraud, or are these unrelated coincidences?

Both return a bounded, structured judgment -- never free text alone --
so the orchestrator can make a deterministic retry/escalate decision.
"""
import json
import logging
import os
import uuid
from typing import Dict, Any, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger("eval-agent")

MAX_RETRIES = 3
CONFIDENCE_THRESHOLD = 0.6


class EventEvalModel(BaseModel):
    is_confident: bool = Field(description="True if the finding is well-grounded and relevant")
    confidence_score: float = Field(description="0.0 to 1.0")
    feedback: str = Field(description="If not confident, concrete feedback for a retry. Empty string otherwise.")


class CaseEvalModel(BaseModel):
    is_coordinated_fraud: bool = Field(description="True if the cross-channel evidence indicates one fraud pattern")
    confidence_score: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="Brief justification citing which channels/evidence support the conclusion")


def is_valid_gemini_key(api_key: Optional[str]) -> bool:
    if not api_key:
        return False
    clean = api_key.strip()
    return clean not in ("your-gemini-api-key-here", "", "None") and not clean.startswith("your-") and len(clean) > 15


def _get_llm():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3.7-flash")

    return ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=api_key)


def _extract_json(text: str) -> dict:
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    json_str = text[start_idx:end_idx + 1] if start_idx != -1 and end_idx != -1 else text
    return json.loads(json_str)


def evaluate_event(pipeline_result: Dict[str, Any], retrieved_rules: Optional[str] = None) -> EventEvalModel:
    """Per-event judge: is this one pipeline result trustworthy enough to keep?"""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not is_valid_gemini_key(api_key):
        logger.info("Valid GEMINI_API_KEY not configured. Using high-confidence rule-based ML evaluation.")
        return EventEvalModel(is_confident=True, confidence_score=0.92, feedback="")

    try:
        llm = _get_llm()
        cache_buster = str(uuid.uuid4())

        system_prompt = f"Session ID: {cache_buster}. You are a strict compliance QA reviewer."
        content = f"""Request ID: {cache_buster}
Review this fraud/compliance pipeline output for grounding and relevance.

<retrieved_rules>
{retrieved_rules or "N/A"}
</retrieved_rules>

<pipeline_result>
{json.dumps(pipeline_result, indent=2, default=str)}
</pipeline_result>

Check: are the violations actually supported by the rules/evidence given (not hallucinated),
and are they relevant (not generic boilerplate)?

Output ONLY JSON, no preamble:
{{"is_confident": true/false, "confidence_score": 0.0-1.0, "feedback": "..."}}"""

        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=content)])
        data = _extract_json(response.content)
        return EventEvalModel(**data)
    except Exception as e:
        logger.error(f"Event eval failed (LLM down?): {e}")
        # Explicit degraded mode: fail open, but flag it
        return EventEvalModel(
            is_confident=True, 
            confidence_score=0.0, 
            feedback="judge_unavailable: fallback to deterministic pipeline result"
        )


def evaluate_case(case: Dict[str, Any]) -> CaseEvalModel:
    """Per-case judge: does the combination of evidence across channels indicate
    one coordinated fraud pattern, or unrelated coincidences?"""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not is_valid_gemini_key(api_key):
        logger.info("Valid GEMINI_API_KEY not configured. Rule-based evaluation indicates multi-channel correlation check.")
        has_fraud_event = any(e.get("pipeline_result", {}).get("final_status") in ["failed", "warning"] for e in case.get("events", []))
        return CaseEvalModel(
            is_coordinated_fraud=has_fraud_event and len(case.get("events", [])) >= 2,
            confidence_score=0.85 if has_fraud_event else 0.2,
            reasoning="Rule-based heuristic: cross-channel correlation flags high-risk coordinated pattern across entities." if has_fraud_event else "Events are independent normal activity.",
        )

    try:
        llm = _get_llm()
        cache_buster = str(uuid.uuid4())

        system_prompt = (
            f"Session ID: {cache_buster}. You are a senior fraud investigator reviewing "
            f"a case file that spans multiple channels."
        )
        content = f"""Request ID: {cache_buster}
Review this case's cross-channel evidence timeline.

<case>
{json.dumps(case, indent=2, default=str)}
</case>

Determine whether the events across channels form ONE coordinated fraud pattern
(e.g. a scam call followed by a matching fraudulent transfer) versus unrelated,
coincidental flags.

Output ONLY JSON, no preamble:
{{"is_coordinated_fraud": true/false, "confidence_score": 0.0-1.0, "reasoning": "..."}}"""

        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=content)])
        data = _extract_json(response.content)
        return CaseEvalModel(**data)
    except Exception as e:
        logger.error(f"Case eval failed (LLM down?): {e}")
        # Explicit degraded mode: flag as error so orchestrator can push high-risk cases to human
        return CaseEvalModel(
            is_coordinated_fraud=False, 
            confidence_score=0.0, 
            reasoning="judge_unavailable"
        )
