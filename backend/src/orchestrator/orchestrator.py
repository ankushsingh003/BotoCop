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
    new_status = case.status  # track what the case becomes

    if risk["should_escalate"]:
        case_eval = eval_agent.evaluate_case(full_case)

        # Check if the judge was unavailable (LLM was down)
        judge_was_down = (case_eval.reasoning == "judge_unavailable")

        if judge_was_down and risk["risk_score"] >= 0.85:
            # Safety policy: when the AI can't decide and risk is high,
            # send to a human analyst. Never auto-clear a high-risk case.
            new_status = CaseStatus.PENDING_REVIEW
            update_case_status(case.case_id, CaseStatus.PENDING_REVIEW, risk_score=risk["risk_score"])
            CASE_STATUS_TRANSITIONS.labels(status=CaseStatus.PENDING_REVIEW.value).inc()
            logger.warning(
                f"Case {case.case_id}: judge unavailable but risk={risk['risk_score']:.2f}, "
                f"routing to PENDING_REVIEW for human analyst"
            )
        elif case_eval.is_coordinated_fraud and case_eval.confidence_score >= CASE_JUDGE_CONFIDENCE_THRESHOLD:
            new_status = CaseStatus.ESCALATED
            update_case_status(case.case_id, CaseStatus.ESCALATED, risk_score=risk["risk_score"])
            CASE_STATUS_TRANSITIONS.labels(status=CaseStatus.ESCALATED.value).inc()
        elif (not case_eval.is_coordinated_fraud) and case_eval.confidence_score >= CASE_JUDGE_CONFIDENCE_THRESHOLD:
            new_status = CaseStatus.CLOSED_CLEARED
            update_case_status(case.case_id, CaseStatus.CLOSED_CLEARED, risk_score=risk["risk_score"])
            CASE_STATUS_TRANSITIONS.labels(status=CaseStatus.CLOSED_CLEARED.value).inc()
        else:
            update_case_status(case.case_id, case.status, risk_score=risk["risk_score"])
    else:
        update_case_status(case.case_id, case.status, risk_score=risk["risk_score"])

    # --- Notification layer (Gap 4): tell someone about escalated cases ---
    # Without this, escalation just changes a DB row that nobody is watching.
    notification_result = None
    customer_warning_result = None
    status_value = new_status.value if isinstance(new_status, CaseStatus) else str(new_status)
    
    if status_value in ("escalated", "pending_review"):
        try:
            from backend.src.notifications.dispatcher import notify_analyst_team, warn_customer
            
            # Get the list of channels involved in this case
            channels_in_case = list(set(
                e.get("channel", "unknown") for e in full_case.get("events", [])
            ))
            
            # Alert the fraud analyst team
            notification_result = notify_analyst_team(
                case_id=str(case.case_id),
                entity_id=case.entity_id,
                risk_score=risk["risk_score"],
                status=status_value,
                channels_involved=channels_in_case,
                summary=f"Cross-channel fraud detected: {', '.join(channels_in_case)} "
                        f"(risk={risk['risk_score']:.2f})",
            )
            
            # Warn the customer if the case is escalated (confirmed threat)
            if status_value == "escalated":
                customer_warning_result = warn_customer(
                    entity_id=case.entity_id,
                    case_id=str(case.case_id),
                    warning_type="scam_warning",
                )
        except Exception as e:
            logger.error(f"Notification dispatch failed (non-blocking): {e}")

    # --- Action layer (Gap 1): decide what to actually DO ---
    # Without this, detection doesn't stop money from moving.
    action_result = None
    try:
        from backend.src.actions.policy_engine import evaluate_action, execute_actions
        case_snapshot = {
            "case_id": str(case.case_id),
            "entity_id": case.entity_id,
            "status": status_value,
            "risk_score": risk["risk_score"],
        }
        event_snapshot = {
            "channel": channel,
            "is_first_time_payee": event_payload.get("is_first_time_payee", False),
        }
        decision = evaluate_action(case_snapshot, event_snapshot)
        if "allow" not in decision.actions:
            # Only execute non-trivial actions (hold, step-up, warn)
            action_result = execute_actions(decision, case.entity_id)
            action_result["policy"] = decision.policy_name
    except Exception as e:
        logger.error(f"Action engine failed (non-blocking): {e}")

    return {
        "case_id": str(case.case_id),
        "channel": channel,
        "pipeline_result": pipeline_result,
        "event_eval": event_eval.model_dump() if event_eval else None,
        "case_risk": risk,
        "case_eval": case_eval.model_dump() if case_eval else None,
        "action_taken": action_result,
        "notification_sent": notification_result is not None,
        "customer_warned": customer_warning_result is not None,
    }
