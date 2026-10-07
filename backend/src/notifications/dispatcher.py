"""
Notification service: makes sure humans actually hear about fraud cases.

The problem this solves: right now, when a case gets escalated at 10:12 AM,
the system changes a status field in a database. But nobody is told. If an
analyst doesn't happen to look at the dashboard in the next 5 hours, Rohan's
48,000 rupees are gone by the time anyone notices.

This module does two things:
1. Alerts the fraud analyst team when a case needs attention
2. Warns the customer directly ("Did you receive a call about a prize?")

It also tracks SLA timers: if no analyst acts on an escalated case within
the configured time window, it auto-escalates to a senior analyst.

In production, the "send" functions would call real APIs (Twilio for SMS,
SendGrid for email, PagerDuty for analyst alerts). Here we simulate them
and persist the notification queue in memory (would be Postgres in prod).
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger("notification-service")


class NotificationChannel(str, Enum):
    """How we reach the person."""
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    DASHBOARD = "dashboard"
    PAGERDUTY = "pagerduty"


class NotificationPriority(str, Enum):
    """How urgent is this notification."""
    LOW = "low"           # informational, can wait
    MEDIUM = "medium"     # should be seen within an hour
    HIGH = "high"         # needs attention within 15 minutes
    CRITICAL = "critical" # someone needs to act RIGHT NOW


class NotificationType(str, Enum):
    """What kind of notification is this."""
    ANALYST_ALERT = "analyst_alert"        # tell the fraud team
    CUSTOMER_WARNING = "customer_warning"  # warn the victim
    SLA_BREACH = "sla_breach"              # nobody acted in time
    CASE_UPDATE = "case_update"            # status changed


# How long an analyst has to act before we escalate
SLA_WINDOWS = {
    NotificationPriority.CRITICAL: timedelta(minutes=5),
    NotificationPriority.HIGH: timedelta(minutes=15),
    NotificationPriority.MEDIUM: timedelta(hours=1),
    NotificationPriority.LOW: timedelta(hours=4),
}


# In-memory notification queue (would be Postgres in production)
_notification_queue: List[Dict[str, Any]] = []
_notification_id_counter = 0


def _next_id() -> str:
    global _notification_id_counter
    _notification_id_counter += 1
    return f"NOTIF-{_notification_id_counter:06d}"


def notify_analyst_team(
    case_id: str,
    entity_id: str, 
    risk_score: float,
    status: str,
    channels_involved: List[str],
    summary: str = "",
) -> Dict[str, Any]:
    """
    Alert the fraud analyst team about a case that needs their attention.
    
    In production this would:
    - Push to PagerDuty if critical
    - Send to the analyst dashboard queue
    - Email the on-call analyst
    - Start an SLA timer
    
    Here we log it and add it to the in-memory queue.
    """
    # Figure out how urgent this is based on case state
    if status == "escalated" and risk_score >= 0.85:
        priority = NotificationPriority.CRITICAL
    elif status == "escalated":
        priority = NotificationPriority.HIGH
    elif status == "pending_review":
        priority = NotificationPriority.HIGH
    else:
        priority = NotificationPriority.MEDIUM
    
    sla_deadline = datetime.now(timezone.utc) + SLA_WINDOWS[priority]
    
    notification = {
        "id": _next_id(),
        "type": NotificationType.ANALYST_ALERT.value,
        "priority": priority.value,
        "case_id": case_id,
        "entity_id": entity_id,
        "risk_score": risk_score,
        "status": status,
        "channels": channels_involved,
        "summary": summary or f"Case {case_id} for entity {entity_id} needs review",
        "sla_deadline": sla_deadline.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "acknowledged": False,
        "acknowledged_by": None,
        "acknowledged_at": None,
    }
    
    _notification_queue.append(notification)
    
    logger.info(
        f"ANALYST ALERT [{priority.value.upper()}]: Case {case_id} "
        f"(entity={entity_id}, risk={risk_score:.2f}, status={status}). "
        f"SLA deadline: {sla_deadline.strftime('%H:%M:%S UTC')}"
    )
    
    # In production: send to PagerDuty, Slack, email, etc.
    # For now we just simulate it
    if priority == NotificationPriority.CRITICAL:
        logger.warning(
            f"PAGERDUTY ALERT would fire for case {case_id} "
            f"(risk={risk_score:.2f}, {len(channels_involved)} channels)"
        )
    
    return notification


def warn_customer(
    entity_id: str,
    case_id: str,
    warning_type: str = "scam_warning",
) -> Dict[str, Any]:
    """
    Send a warning directly to the customer.
    
    For example: "Did you receive a call about a lottery prize?
    This is a known scam pattern. Do not make any payments."
    
    In production this would call Twilio (SMS) or the bank's
    push notification API. Here we simulate it.
    """
    # Pick the right warning message based on what we detected
    messages = {
        "scam_warning": (
            "SECURITY ALERT: We detected unusual activity linked to your account. "
            "If you received a call about a prize, refund, or fee, it may be a scam. "
            "Do NOT make any payments. Call us at 1800-XXX-XXXX to verify."
        ),
        "phishing_warning": (
            "SECURITY ALERT: A suspicious email was sent to your address. "
            "Do not click any links or share personal information. "
            "Contact us to verify any requests."
        ),
        "transaction_warning": (
            "SECURITY ALERT: A transaction from your account has been "
            "flagged for review. If you did not authorize this, "
            "call us immediately at 1800-XXX-XXXX."
        ),
    }
    
    message = messages.get(warning_type, messages["scam_warning"])
    
    notification = {
        "id": _next_id(),
        "type": NotificationType.CUSTOMER_WARNING.value,
        "priority": NotificationPriority.HIGH.value,
        "entity_id": entity_id,
        "case_id": case_id,
        "message": message,
        "delivery_channel": "sms",  # would be configurable per customer
        "created_at": datetime.now(timezone.utc).isoformat(),
        "delivered": True,  # simulated
    }
    
    _notification_queue.append(notification)
    
    logger.info(
        f"CUSTOMER WARNING sent to {entity_id}: {warning_type} "
        f"(case={case_id})"
    )
    
    return notification


def check_sla_breaches() -> List[Dict[str, Any]]:
    """
    Check for notifications where the SLA deadline has passed and
    no analyst has acknowledged them. Returns a list of breached
    notifications that need escalation.
    
    In production, this would run on a cron job (e.g. every minute)
    and auto-escalate to a senior analyst or manager.
    """
    now = datetime.now(timezone.utc)
    breaches = []
    
    for notif in _notification_queue:
        if notif["type"] != NotificationType.ANALYST_ALERT.value:
            continue
        if notif.get("acknowledged"):
            continue
            
        deadline = datetime.fromisoformat(notif["sla_deadline"])
        if now > deadline:
            breach = {
                "id": _next_id(),
                "type": NotificationType.SLA_BREACH.value,
                "priority": NotificationPriority.CRITICAL.value,
                "original_notification_id": notif["id"],
                "case_id": notif["case_id"],
                "entity_id": notif["entity_id"],
                "sla_deadline": notif["sla_deadline"],
                "breach_duration_minutes": int((now - deadline).total_seconds() / 60),
                "created_at": now.isoformat(),
            }
            breaches.append(breach)
            logger.warning(
                f"SLA BREACH: Case {notif['case_id']} has been waiting "
                f"{breach['breach_duration_minutes']} minutes past deadline"
            )
    
    return breaches


def acknowledge_notification(notification_id: str, analyst_id: str) -> bool:
    """
    Mark a notification as acknowledged by an analyst.
    Stops the SLA timer for this notification.
    """
    for notif in _notification_queue:
        if notif["id"] == notification_id:
            notif["acknowledged"] = True
            notif["acknowledged_by"] = analyst_id
            notif["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(
                f"Notification {notification_id} acknowledged by {analyst_id}"
            )
            return True
    return False


def get_pending_notifications(priority: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get all unacknowledged analyst notifications, optionally filtered by priority.
    This is what the analyst dashboard polls to show the alert queue.
    """
    pending = [
        n for n in _notification_queue
        if n["type"] == NotificationType.ANALYST_ALERT.value
        and not n.get("acknowledged")
    ]
    
    if priority:
        pending = [n for n in pending if n["priority"] == priority]
    
    # Sort by priority (critical first) then by creation time (oldest first)
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    pending.sort(key=lambda n: (priority_order.get(n["priority"], 99), n["created_at"]))
    
    return pending


def get_notification_stats() -> Dict[str, Any]:
    """Summary stats for the dashboard."""
    analyst_alerts = [n for n in _notification_queue if n["type"] == NotificationType.ANALYST_ALERT.value]
    customer_warnings = [n for n in _notification_queue if n["type"] == NotificationType.CUSTOMER_WARNING.value]
    unacked = [n for n in analyst_alerts if not n.get("acknowledged")]
    
    return {
        "total_notifications": len(_notification_queue),
        "analyst_alerts": len(analyst_alerts),
        "customer_warnings": len(customer_warnings),
        "pending_acknowledgement": len(unacked),
        "sla_breaches": len(check_sla_breaches()),
    }
