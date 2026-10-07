"""
Action policies: what the system actually DOES when it detects fraud.

The detection engine (orchestrator, pipelines, judges) figures out
that something looks suspicious and updates the case status. But
changing a database row doesn't stop money from moving. This module
bridges that gap.

Each policy is a simple rule: "if the case looks like X, do Y."
Policies are plain config, not LLM calls, because action decisions
must be fast (< 50ms) and predictable. You don't want a language
model deciding whether to freeze someone's bank account.

Example policies:
  - ESCALATED case + outgoing transfer = hold the transfer
  - High risk score + new payee = require OTP verification  
  - Any active case + large transfer = warn the customer first
  
In a real bank, this module would call the payment gateway's
hold/release API. Here we simulate the decision and log it.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

logger = logging.getLogger("action-engine")


# These are the things we can actually do to protect the customer.
# Each action has a name and a human-readable description.
AVAILABLE_ACTIONS = {
    "hold": "Temporarily hold the transaction for manual review",
    "step_up": "Require additional verification (OTP, biometric, callback)",
    "warn_customer": "Send the customer a warning message about a potential scam",
    "temporary_limit": "Lower the customer's transaction limit until case is resolved",
    "allow": "Let the transaction through (no action needed)",
}


# Action policies: simple if-then rules that map case state to actions.
# The key idea: policies live in config, not in the LLM, because
# action decisions must be fast and predictable.
ACTION_POLICIES = [
    {
        "name": "escalated_transfer_hold",
        "description": "If we already flagged this customer as a fraud victim "
                       "and they're trying to send money, hold it",
        "condition": lambda case, event: (
            case.get("status") == "escalated" 
            and event.get("channel") == "transaction"
        ),
        "actions": ["hold", "warn_customer"],
        "priority": 1,
    },
    {
        "name": "pending_review_step_up",
        "description": "Case is waiting for human review and the customer is "
                       "doing something risky, ask for extra verification",
        "condition": lambda case, event: (
            case.get("status") == "pending_review"
            and event.get("channel") == "transaction"
        ),
        "actions": ["step_up"],
        "priority": 2,
    },
    {
        "name": "high_risk_new_payee",
        "description": "High risk score and the customer is paying someone new, "
                       "require OTP before we let it through",
        "condition": lambda case, event: (
            case.get("risk_score", 0) >= 0.7
            and event.get("is_first_time_payee", False)
        ),
        "actions": ["step_up", "warn_customer"],
        "priority": 3,
    },
    {
        "name": "medium_risk_warning",
        "description": "Moderate risk, just warn the customer so they can "
                       "decide for themselves",
        "condition": lambda case, event: (
            0.4 <= case.get("risk_score", 0) < 0.7
            and event.get("channel") == "transaction"
        ),
        "actions": ["warn_customer"],
        "priority": 4,
    },
]


class ActionDecision:
    """The result of running an event through the action policies."""
    
    def __init__(self, actions: List[str], policy_name: str, reason: str, priority: int):
        self.actions = actions
        self.policy_name = policy_name
        self.reason = reason
        self.priority = priority
        self.decided_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> dict:
        return {
            "actions": self.actions,
            "policy_name": self.policy_name,
            "reason": self.reason,
            "priority": self.priority,
            "decided_at": self.decided_at.isoformat(),
        }


# In-memory log of actions taken (in production this would be a DB table)
_action_log: List[Dict[str, Any]] = []


def evaluate_action(case_data: Optional[Dict], event_data: Dict) -> ActionDecision:
    """
    Run the event through all action policies and return the highest
    priority action that matches. This is the function the payment
    gateway would call before authorizing a transfer.
    
    Designed to be very fast (no LLM, no network calls, just dict lookups)
    so it can sit in the transaction authorization path.
    """
    if case_data is None:
        # No open case for this customer, let it through
        return ActionDecision(
            actions=["allow"],
            policy_name="no_case",
            reason="No active fraud case for this customer",
            priority=99,
        )
    
    # Check each policy in priority order
    matched_policies = []
    for policy in ACTION_POLICIES:
        try:
            if policy["condition"](case_data, event_data):
                matched_policies.append(policy)
        except Exception as e:
            logger.warning(f"Policy {policy['name']} check failed: {e}")
    
    if not matched_policies:
        return ActionDecision(
            actions=["allow"],
            policy_name="no_match",
            reason="No action policy triggered for this case state",
            priority=99,
        )
    
    # Use the highest priority (lowest number) matching policy
    best = min(matched_policies, key=lambda p: p["priority"])
    decision = ActionDecision(
        actions=best["actions"],
        policy_name=best["name"],
        reason=best["description"],
        priority=best["priority"],
    )
    
    # Log the action for audit trail
    log_entry = {
        "case_id": case_data.get("case_id"),
        "entity_id": case_data.get("entity_id"),
        "event_channel": event_data.get("channel"),
        **decision.to_dict(),
    }
    _action_log.append(log_entry)
    logger.info(
        f"Action decision for {case_data.get('entity_id')}: "
        f"{decision.actions} (policy={decision.policy_name})"
    )
    
    return decision


def execute_actions(decision: ActionDecision, entity_id: str) -> Dict[str, Any]:
    """
    Actually carry out the actions. In a real bank this would call
    external APIs (payment gateway, SMS service, etc). Here we
    simulate and log what would happen.
    """
    results = {}
    
    for action in decision.actions:
        if action == "hold":
            # In production: call payment_gateway.hold_transaction(txn_id)
            results["hold"] = {
                "executed": True,
                "message": f"Transaction held for entity {entity_id} pending manual review",
            }
            logger.info(f"HOLD executed for {entity_id}")
            
        elif action == "step_up":
            # In production: call auth_service.require_otp(entity_id)
            results["step_up"] = {
                "executed": True,
                "message": f"OTP/biometric verification required for entity {entity_id}",
            }
            logger.info(f"STEP-UP AUTH required for {entity_id}")
            
        elif action == "warn_customer":
            # In production: call notification_service.send_warning(entity_id, message)
            results["warn_customer"] = {
                "executed": True,
                "message": f"Warning sent to entity {entity_id}: "
                           f"'We detected unusual activity on your account. "
                           f"If you received a call about a prize or refund, do not pay.'",
            }
            logger.info(f"WARNING sent to {entity_id}")
            
        elif action == "temporary_limit":
            # In production: call account_service.set_temp_limit(entity_id, limit=10000)
            results["temporary_limit"] = {
                "executed": True,
                "message": f"Transaction limit temporarily reduced for entity {entity_id}",
            }
            logger.info(f"TEMP LIMIT set for {entity_id}")
            
        elif action == "allow":
            results["allow"] = {"executed": True, "message": "Transaction allowed"}
    
    return results


def get_action_log() -> List[Dict[str, Any]]:
    """Return all action decisions taken (for the dashboard/audit trail)."""
    return list(_action_log)
