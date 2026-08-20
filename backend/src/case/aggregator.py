"""
Cross-channel risk aggregation.

Combines the severities of every event in a case into one cumulative
risk score, and decides whether the case should escalate to the
per-case eval agent. Deliberately simple and explainable (weighted
sum, not a black-box model) so the eval agent -- and you, in an
interview -- can reason about exactly why a case crossed the threshold.
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger("case-aggregator")

SEVERITY_WEIGHTS = {"low": 0.1, "medium": 0.35, "high": 0.7}

# A single high-severity hit in ONE channel is a strong signal but
# shouldn't alone trigger cross-channel escalation -- that requires
# evidence from more than one channel, which is the point of this layer.
ESCALATION_THRESHOLD = 0.6
MIN_CHANNELS_FOR_ESCALATION = 2


def compute_case_risk(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    events: [{"channel": str, "pipeline_result": {"violations": [...]}}]
    Each violation is expected to carry a "severity" of low/medium/high,
    matching the normalized schema your existing merge_results_node produces.
    """
    channels_seen = set()
    max_weight = 0.0

    for event in events:
        channels_seen.add(event["channel"])
        violations = (event.get("pipeline_result") or {}).get("violations", [])
        for v in violations:
            w = SEVERITY_WEIGHTS.get((v.get("severity") or "low").lower(), 0.1)
            max_weight = max(max_weight, w)

    # Cross-channel presence bumps the score: one hit each in 2 channels
    # is more suspicious than 2 hits in a single channel.
    channel_bonus = 0.15 * (len(channels_seen) - 1) if channels_seen else 0.0
    risk_score = min(1.0, max_weight + channel_bonus)

    should_escalate = (
        risk_score >= ESCALATION_THRESHOLD
        and len(channels_seen) >= MIN_CHANNELS_FOR_ESCALATION
    )

    return {
        "risk_score": round(risk_score, 3),
        "channels_seen": sorted(channels_seen),
        "should_escalate": should_escalate,
    }
