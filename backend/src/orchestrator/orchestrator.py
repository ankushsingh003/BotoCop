"""
Single entry point for any incoming channel event.

Flow: link to a case -> run the channel's specialist pipeline inside a
bounded eval/retry loop -> persist the result -> recompute cross-channel
risk -> if 2+ channels now have evidence, ask the case-level judge
whether this is coordinated fraud.
"""
import logging
import time
from typing import Dict, Any, Callable

from backend.src.case.linker import link_event_to_case
from backend.src.case.store import append_event, get_case_with_events, update_case_status
from backend.src.case.aggregator import compute_case_risk
from backend.src.case.models import CaseStatus
from backend.src.datalake.writer import archive_event
from backend.src.monitoring.metrics import (
    EVENTS_PROCESSED,
    VIOLATIONS_DETECTED,
    PIPELINE_DURATION_SECONDS,
    EVAL_RETRIES,
    CASE_STATUS_TRANSITIONS,
    CASE_RISK_SCORE,
    CALL_ML_FRAUD_PROBABILITY,
    CALL_ML_RISK_LEVEL,
)

from backend.src.orchestrator import eval_agent as default_eval_agent
from backend.src.pipelines.transaction_fraud.workflow import run_transaction_fraud_pipeline
from backend.src.pipelines.call_fraud.workflow import run_call_fraud_pipeline
from backend.src.pipelines.text_fraud.workflow import run_text_fraud_pipeline

logger = logging.getLogger("orchestrator")

PIPELINES: Dict[str, Callable] = {
    "transaction": run_transaction_fraud_pipeline,
    "call": run_call_fraud_pipeline,
    "text": run_text_fraud_pipeline,
}

PIPELINE_REQUIRES_EVAL: Dict[str, bool] = {
    "transaction": True,
    "call": True,
    "text": True,
}

MAX_RETRIES = 3
CASE_JUDGE_CONFIDENCE_THRESHOLD = 0.6


def handle_event(
    channel: str,
    event_payload: Dict[str, Any],
    eval_agent=default_eval_agent,
) -> Dict[str, Any]:
    if channel not in PIPELINES:
        raise ValueError(f"No pipeline registered for channel '{channel}'")

    case = link_event_to_case(channel, event_payload)
    logger.info(f"Event routed: channel={channel}, case={case.case_id}, entity={case.entity_id}")

    pipeline_fn = PIPELINES[channel]
    requires_eval = PIPELINE_REQUIRES_EVAL.get(channel, True)
    retry_feedback = None
    pipeline_result: Dict[str, Any] = {}
    event_eval = None

    pipeline_start = time.perf_counter()
    if requires_eval:
        for attempt in range(1, MAX_RETRIES + 1):
            pipeline_result = pipeline_fn(event_payload, retry_feedback)
            event_eval = eval_agent.evaluate_event(
                pipeline_result, retrieved_rules=pipeline_result.get("rag_sources")
            )
            if event_eval.is_confident or attempt == MAX_RETRIES:
                if not event_eval.is_confident:
                    logger.warning(f"Exhausted {MAX_RETRIES} retries without reaching confidence; proceeding anyway.")
                break
            logger.info(f"Attempt {attempt}: eval not confident (score={event_eval.confidence_score}), retrying")
            EVAL_RETRIES.labels(channel=channel).inc()
            retry_feedback = event_eval.feedback
    else:
        pipeline_result = pipeline_fn(event_payload, retry_feedback)
    PIPELINE_DURATION_SECONDS.labels(channel=channel).observe(time.perf_counter() - pipeline_start)

    EVENTS_PROCESSED.labels(channel=channel, final_status=pipeline_result.get("final_status", "unknown")).inc()
    for v in pipeline_result.get("violations", []):
        VIOLATIONS_DETECTED.labels(channel=channel, severity=v.get("severity", "unknown")).inc()

    if channel == "call" and "ml_score" in pipeline_result:
        ml_score = pipeline_result["ml_score"]
        if "fraud_probability" in ml_score:
            CALL_ML_FRAUD_PROBABILITY.observe(ml_score["fraud_probability"])
        if "risk_level" in ml_score:
            CALL_ML_RISK_LEVEL.labels(risk_level=ml_score["risk_level"]).inc()


    append_event(case.case_id, channel=channel, pipeline_result=pipeline_result)
    archive_event(channel, event_payload, pipeline_result, case_id=str(case.case_id))

    full_case = get_case_with_events(case.case_id)
    risk = compute_case_risk(full_case["events"])
    CASE_RISK_SCORE.observe(risk["risk_score"])
    case_eval = None

    if risk["should_escalate"]:
        case_eval = eval_agent.evaluate_case(full_case)
        if case_eval.is_coordinated_fraud and case_eval.confidence_score >= CASE_JUDGE_CONFIDENCE_THRESHOLD:
            update_case_status(case.case_id, CaseStatus.ESCALATED, risk_score=risk["risk_score"])
            CASE_STATUS_TRANSITIONS.labels(status=CaseStatus.ESCALATED.value).inc()
        elif (not case_eval.is_coordinated_fraud) and case_eval.confidence_score >= CASE_JUDGE_CONFIDENCE_THRESHOLD:
            update_case_status(case.case_id, CaseStatus.CLOSED_CLEARED, risk_score=risk["risk_score"])
            CASE_STATUS_TRANSITIONS.labels(status=CaseStatus.CLOSED_CLEARED.value).inc()
        else:
            update_case_status(case.case_id, case.status, risk_score=risk["risk_score"])
    else:
        update_case_status(case.case_id, case.status, risk_score=risk["risk_score"])

    return {
        "case_id": str(case.case_id),
        "channel": channel,
        "pipeline_result": pipeline_result,
        "event_eval": event_eval.model_dump() if event_eval else None,
        "case_risk": risk,
        "case_eval": case_eval.model_dump() if case_eval else None,
    }
