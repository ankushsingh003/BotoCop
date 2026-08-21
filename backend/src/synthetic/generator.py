"""
Synthetic data generator.

We don't have access to real banking/call data, so this generates
realistic-shaped normal and fraudulent events for two purposes:
  1. Training the transaction anomaly detector on what "normal" looks like
     (unsupervised -- no fraud labels needed at train time).
  2. End-to-end testing, including a "correlated fraud case" generator that
     produces a scam call immediately followed by a matching fraudulent
     transfer -- the exact pattern the cross-channel case layer exists to
     catch, and something a single-channel pipeline could never catch alone.
"""
import random
import uuid
from datetime import datetime, timedelta, timezone

LOW_RISK_COUNTRIES = ["US", "CA", "UK", "DE", "FR", "AU"]
HIGH_RISK_COUNTRIES = ["XX", "YY", "ZZ"]

NORMAL_CALL_TRANSCRIPTS = [
    "Hey, just calling to confirm our lunch plans for Saturday.",
    "Hi, this is regarding your recent order, it shipped yesterday and should arrive Friday.",
    "Good morning, following up on the meeting notes from yesterday.",
    "Hello, just checking in to see how you're doing this week.",
    "Hi, your dentist appointment is confirmed for next Tuesday at 2pm.",
]

FRAUD_CALL_TRANSCRIPTS = [
    "This is your bank's security department, we've detected suspicious activity, "
    "please confirm your identity by wiring funds to our verification account immediately.",
    "Hi, this is IT support, your computer has a virus, we need you to purchase gift "
    "cards right away to fix it.",
    "This is the tax authority, you owe back taxes and must wire transfer payment today "
    "or a warrant will be issued.",
    "Congratulations, you've won a prize, but you must first send a processing fee by "
    "wire transfer to claim it.",
]


def _new_entity_id() -> str:
    return f"acct_{uuid.uuid4().hex[:8]}"


def generate_normal_transaction(entity_id: str = None) -> dict:
    hour = random.randint(7, 22)
    ts = datetime.now(timezone.utc).replace(hour=hour, minute=random.randint(0, 59), second=0, microsecond=0)
    return {
        "account_id": entity_id or _new_entity_id(),
        "amount": round(random.uniform(10, 500), 2),
        "currency": "USD",
        "merchant": random.choice(["Grocery Mart", "Coffee Shop", "Gas Station", "Electric Co"]),
        "txn_type": "purchase",
        "country": random.choice(LOW_RISK_COUNTRIES),
        "is_new_payee": False,
        "timestamp": ts.isoformat(),
    }


def generate_fraud_transaction(entity_id: str = None) -> dict:
    odd_hour = random.choice([1, 2, 3, 23])
    ts = datetime.now(timezone.utc).replace(hour=odd_hour, minute=random.randint(0, 59))
    return {
        "account_id": entity_id or _new_entity_id(),
        "amount": round(random.uniform(3000, 15000), 2),
        "currency": "USD",
        "merchant": f"New Payee {uuid.uuid4().hex[:6]}",
        "txn_type": "wire_transfer",
        "country": random.choice(HIGH_RISK_COUNTRIES),
        "is_new_payee": True,
        "timestamp": ts.isoformat(),
    }


def generate_normal_call(entity_id: str = None) -> dict:
    return {
        "linked_account_id": entity_id or _new_entity_id(),
        "transcript": random.choice(NORMAL_CALL_TRANSCRIPTS),
        "duration_seconds": random.randint(30, 300),
    }


def generate_fraud_call(entity_id: str = None) -> dict:
    return {
        "linked_account_id": entity_id or _new_entity_id(),
        "transcript": random.choice(FRAUD_CALL_TRANSCRIPTS),
        "duration_seconds": random.randint(120, 600),
    }


NORMAL_TEXT_MESSAGES = [
    {"subject": "Lunch tomorrow?", "body": "Hey, are we still on for lunch tomorrow at noon?", "sender": "friend@gmail.com"},
    {"subject": "Your order has shipped", "body": "Your order #48213 has shipped and will arrive Friday.", "sender": "orders@retailstore.com"},
    {"subject": "Meeting notes", "body": "Attached are the notes from today's planning meeting.", "sender": "colleague@company.com"},
]

FRAUD_TEXT_MESSAGES = [
    {
        "subject": "Urgent: Verify your account now",
        "body": "Your account has been suspended. Click here immediately and enter your password "
                "and OTP to restore access or your account will be permanently closed within 24 hours.",
        "sender": "security-alert@bank-verify-support.com",
    },
    {
        "subject": "You have won!",
        "body": "Congratulations, you've won a $1000 gift card. Reply with your card number and "
                "CVV to claim your prize before it expires today.",
        "sender": "rewards@prize-notification.net",
    },
]


def generate_normal_text(entity_id: str = None) -> dict:
    template = random.choice(NORMAL_TEXT_MESSAGES)
    return {
        "linked_account_id": entity_id or _new_entity_id(),
        "sender_email": template["sender"],
        "subject": template["subject"],
        "body": template["body"],
    }


def generate_fraud_text(entity_id: str = None) -> dict:
    template = random.choice(FRAUD_TEXT_MESSAGES)
    return {
        "linked_account_id": entity_id or _new_entity_id(),
        "sender_email": template["sender"],
        "subject": template["subject"],
        "body": template["body"],
    }


def generate_correlated_fraud_case() -> tuple:
    """A scam call immediately followed by a matching fraudulent transfer,
    same entity_id -- returns (call_event, transaction_event)."""
    entity_id = _new_entity_id()
    return generate_fraud_call(entity_id), generate_fraud_transaction(entity_id)


def generate_training_dataset(n: int = 500) -> list:
    """Normal-only transactions for fitting the anomaly detector."""
    return [generate_normal_transaction() for _ in range(n)]
