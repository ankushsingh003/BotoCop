import json
import logging
import os
import re
import uuid
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from backend.src.pipelines.transaction_fraud.state import TransactionFraudState

logger = logging.getLogger("transaction-fraud")


def summarize_transaction(transaction: Dict[str, Any]) -> str:
    """Turn a raw transaction record into a short natural-language summary --
    RAG retrieves against text, so the numeric record needs to become text
    first, the same way the video pipeline turns frames/audio into a
    transcript before retrieval."""
    parts = []
    if "amount" in transaction:
        currency = transaction.get("currency", "")
        parts.append(f"a {currency} {transaction['amount']} transaction")
    if "txn_type" in transaction:
        parts.append(f"of type {transaction['txn_type']}")
    if "merchant" in transaction:
        parts.append(f"to merchant/payee '{transaction['merchant']}'")
    if "country" in transaction:
        parts.append(f"originating from {transaction['country']}")
    if transaction.get("is_new_payee"):
        parts.append("to a first-time payee")
    return " ".join(parts) or "an unspecified banking transaction"


class TransactionViolationModel(BaseModel):
    category: str = Field(description="e.g. Structuring, Unusual_Velocity, High_Risk_Corridor, AML_Threshold")
    description: str = Field(description="Specific finding based on the transaction and guidelines")
    severity: str = Field(description="low, medium, or high")
    suggestion: str = Field(description="Recommended action, e.g. hold for manual review")


class TransactionAuditModel(BaseModel):
    violations: List[TransactionViolationModel] = Field(default_factory=list)
    final_status: str = Field(description="success, warning, or failed")


def retrieve_rules_node(state: TransactionFraudState) -> Dict[str, Any]:
    """Pull relevant finance compliance/AML rules for this transaction from
    the existing RAG retriever, reusing the 'finance' rulebook collection
    (rules_finance.pdf) already indexed for the video pipeline."""
    from backend.src.rag.retriever import RuleRetriever

    transaction = state.get("transaction") or {}
    query = summarize_transaction(transaction)

    logger.info(f"Retrieving finance rules for: {query}")
    try:
        retriever = RuleRetriever()
        retrieved_rules = retriever.retrieve(query, domain="finance")
        rag_sources = re.findall(r"\[Source: ([^\]]+)\]", retrieved_rules)
    except Exception as e:
        logger.error(f"Failed to retrieve finance rules: {e}")
        retrieved_rules = "No matching rules found."
        rag_sources = []

    return {"retrieved_rules": retrieved_rules, "rag_sources": rag_sources}


def audit_transaction_node(state: TransactionFraudState) -> Dict[str, Any]:
    """Gemini LLM audits the transaction against retrieved rules -- same
    JSON-schema-enforced pattern as the video pipeline's auditor node."""
    transaction = state.get("transaction") or {}
    retrieved_rules = state.get("retrieved_rules") or "No rules retrieved."
    retry_feedback = state.get("retry_feedback")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash")

    if not api_key:
        logger.warning("GEMINI_API_KEY not set. Using rule-based fallback for transaction fraud audit.")
        return {"violations": [], "final_status": "success"}

    cache_buster = str(uuid.uuid4())
    system_prompt = (
        f"Session ID: {cache_buster}. You are a Professional Transaction Fraud "
        f"Auditor specializing in AML and banking compliance."
    )
    feedback_block = (
        f"\n<prior_attempt_feedback>\n{retry_feedback}\n</prior_attempt_feedback>\n"
        if retry_feedback else ""
    )

    content = f"""Request ID: {cache_buster}
Analyze the following transaction against the retrieved compliance guidelines.

<guidelines>
{retrieved_rules}
</guidelines>

<transaction>
{json.dumps(transaction, indent=2, default=str)}
</transaction>
{feedback_block}
You MUST output ONLY a valid JSON object matching this schema, no preamble or markdown:
{{
    "violations": [
        {{
            "category": "Structuring/Unusual_Velocity/High_Risk_Corridor/AML_Threshold/General",
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
        report = TransactionAuditModel(**data)

        violations = [v.model_dump() for v in report.violations]
        return {"violations": violations, "final_status": report.final_status}
    except Exception as e:
        logger.error(f"Transaction audit failed: {e}")
        return {"error": [str(e)], "violations": [], "final_status": "failed"}
