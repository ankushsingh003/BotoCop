"""
Golden dataset: labeled examples for offline evaluation of each
pipeline. Not drawn from live traffic -- purpose-built, hand-reviewable
examples with a known correct answer, so a prompt or model change can be
checked for regressions before it ever reaches production.

Deliberately includes borderline cases (a large-but-legitimate purchase,
a call that mentions a bank without being a scam) alongside obvious
ones. An all-obvious dataset would let a pipeline that's just
keyword-matching score perfectly and hide that it isn't actually
reasoning about the content.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GoldenExample:
    id: str
    channel: str
    payload: Dict[str, Any]
    is_fraud: bool  # ground truth
    notes: str = ""


TRANSACTION_EXAMPLES: List[GoldenExample] = [
    GoldenExample(
        id="txn_001",
        channel="transaction",
        payload={
            "account_id": "acct_gold_1", "amount": 9500, "currency": "USD",
            "merchant": "New Payee X", "txn_type": "wire_transfer",
            "country": "XX", "is_new_payee": True,
            "timestamp": "2026-08-05T02:14:00+00:00",
        },
        is_fraud=True,
        notes="Large wire, new payee, high-risk country, odd hour -- clear signal.",
    ),
    GoldenExample(
        id="txn_002",
        channel="transaction",
        payload={
            "account_id": "acct_gold_2", "amount": 45, "currency": "USD",
            "merchant": "Coffee Shop", "txn_type": "purchase",
            "country": "US", "is_new_payee": False,
            "timestamp": "2026-08-05T14:30:00+00:00",
        },
        is_fraud=False,
        notes="Ordinary small daytime purchase.",
    ),
    GoldenExample(
        id="txn_003",
        channel="transaction",
        payload={
            "account_id": "acct_gold_3", "amount": 4200, "currency": "USD",
            "merchant": "Home Renovation Co", "txn_type": "purchase",
            "country": "US", "is_new_payee": True,
            "timestamp": "2026-08-05T11:00:00+00:00",
        },
        is_fraud=False,
        notes="BORDERLINE: large + new payee, but daytime, domestic, plausible contractor "
              "payment. Tests against pure amount/new-payee keyword-matching.",
    ),
    GoldenExample(
        id="txn_004",
        channel="transaction",
        payload={
            "account_id": "acct_gold_4", "amount": 12000, "currency": "USD",
            "merchant": "New Payee Y", "txn_type": "wire_transfer",
            "country": "YY", "is_new_payee": True,
            "timestamp": "2026-08-05T03:47:00+00:00",
        },
        is_fraud=True,
        notes="Very large wire, new payee, high-risk country, 3:47am.",
    ),
    GoldenExample(
        id="txn_005",
        channel="transaction",
        payload={
            "account_id": "acct_gold_5", "amount": 250, "currency": "USD",
            "merchant": "Electric Co", "txn_type": "purchase",
            "country": "CA", "is_new_payee": False,
            "timestamp": "2026-08-05T09:15:00+00:00",
        },
        is_fraud=False,
        notes="Routine recurring bill payment.",
    ),
]

CALL_EXAMPLES: List[GoldenExample] = [
    GoldenExample(
        id="call_001",
        channel="call",
        payload={
            "linked_account_id": "acct_gold_6",
            "transcript": "This is your bank's security department, we've detected suspicious "
                          "activity, please confirm your identity by wiring funds to our "
                          "verification account immediately.",
        },
        is_fraud=True,
        notes="Classic bank-impersonation + urgency + wire-transfer request.",
    ),
    GoldenExample(
        id="call_002",
        channel="call",
        payload={
            "linked_account_id": "acct_gold_7",
            "transcript": "Hi, your dentist appointment is confirmed for next Tuesday at 2pm.",
        },
        is_fraud=False,
        notes="Ordinary appointment reminder.",
    ),
    GoldenExample(
        id="call_003",
        channel="call",
        payload={
            "linked_account_id": "acct_gold_8",
            "transcript": "Hi, this is your bank calling to let you know your new debit card "
                          "has shipped and should arrive within 5-7 business days.",
        },
        is_fraud=False,
        notes="BORDERLINE: mentions 'bank' but is a routine notification, no urgency, "
              "no request for credentials/money. Tests against keyword-only detection.",
    ),
    GoldenExample(
        id="call_004",
        channel="call",
        payload={
            "linked_account_id": "acct_gold_9",
            "transcript": "This is the tax authority, you owe back taxes and must wire "
                          "transfer payment today or a warrant will be issued.",
        },
        is_fraud=True,
        notes="Government impersonation + threat + urgent payment demand.",
    ),
]

TEXT_EXAMPLES: List[GoldenExample] = [
    GoldenExample(
        id="text_001",
        channel="text",
        payload={
            "linked_account_id": "acct_gold_10",
            "sender_email": "security-alert@bank-verify-support.com",
            "subject": "Urgent: Verify your account now",
            "body": "Your account has been suspended. Click here immediately and enter "
                    "your password and OTP to restore access or it will be closed within 24 hours.",
        },
        is_fraud=True,
        notes="Urgency + credential harvest + suspicious sender domain.",
    ),
    GoldenExample(
        id="text_002",
        channel="text",
        payload={
            "linked_account_id": "acct_gold_11",
            "sender_email": "orders@retailstore.com",
            "subject": "Your order has shipped",
            "body": "Your order #48213 has shipped and will arrive Friday.",
        },
        is_fraud=False,
        notes="Ordinary shipping notification.",
    ),
    GoldenExample(
        id="text_003",
        channel="text",
        payload={
            "linked_account_id": "acct_gold_12",
            "sender_email": "notifications@realbank.com",
            "subject": "Your statement is ready",
            "body": "Your monthly statement is now available in online banking. Log in "
                    "through the app or website as usual to view it.",
        },
        is_fraud=False,
        notes="BORDERLINE: bank-related and asks the user to log in, but no link, no "
              "urgency, no credential request -- routine notification, not phishing.",
    ),
]

ALL_EXAMPLES: List[GoldenExample] = TRANSACTION_EXAMPLES + CALL_EXAMPLES + TEXT_EXAMPLES
