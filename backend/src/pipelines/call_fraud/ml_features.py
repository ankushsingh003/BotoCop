import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field

# High risk keyword lexicons for call vishing / scam detection
URGENCY_KEYWORDS = [
    "immediately", "immediate", "urgent", "urgently", "block", "blocked", "deactivate",
    "suspended", "suspension", "police", "jail", "court", "warrant", "legal action",
    "arrest", "expire", "expiration", "terminate", "penalty", "fine", "freeze"
]

OTP_CREDENTIAL_KEYWORDS = [
    "otp", "one time password", "verification code", "cvv", "pin", "password",
    "card number", "expiry date", "secret code", "auth code"
]

IMPERSONATION_KEYWORDS = [
    "bank", "rbi", "reserve bank", "police", "cyber crime", "i4c", "cbi", "customs",
    "fedex", "courier", "trai", "telecom department", "support desk", "security team",
    "state bank", "hdfc", "icici", "axis", "sbi"
]

FINANCIAL_DEMAND_KEYWORDS = [
    "transfer", "wire", "deposit", "gift card", "upi", "google pay", "phonepe",
    "paytm", "refund", "account number", "beneficiary", "amount", "charge", "payment"
]


class CallFeatures(BaseModel):
    urgency_score: float = Field(description="Normalized urgency/fear tactic frequency (0 to 1)")
    otp_request_detected: int = Field(description="1 if OTP/credential request detected, else 0")
    impersonation_score: float = Field(description="Normalized authority impersonation indicator (0 to 1)")
    financial_demand_score: float = Field(description="Normalized financial transfer/payment demand score (0 to 1)")
    is_spoof_suspected: int = Field(description="1 if caller phone number pattern suggests spoofing, else 0")
    call_duration_seconds: float = Field(description="Call duration in seconds")
    complaint_history_count: int = Field(description="Number of prior complaints linked to caller phone")
    off_hours_call: int = Field(description="1 if call placed outside standard hours (9am-6pm)")

    def to_feature_vector(self) -> List[float]:
        """Convert to ordered numerical vector for ML model input."""
        return [
            self.urgency_score,
            float(self.otp_request_detected),
            self.impersonation_score,
            self.financial_demand_score,
            float(self.is_spoof_suspected),
            self.call_duration_seconds / 600.0,  # normalized duration (up to ~10 mins)
            float(self.complaint_history_count),
            float(self.off_hours_call),
        ]

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "urgency_score",
            "otp_request_detected",
            "impersonation_score",
            "financial_demand_score",
            "is_spoof_suspected",
            "call_duration_normalized",
            "complaint_history_count",
            "off_hours_call",
        ]


def extract_call_features(call_event: Dict[str, Any]) -> CallFeatures:
    """
    Automated feature extraction from a call event and transcript.
    """
    transcript = call_event.get("transcript", "").lower()
    caller_phone = call_event.get("caller_phone") or call_event.get("phone_number") or ""
    duration = float(call_event.get("duration_seconds", 0) or call_event.get("duration", 0))
    complaints = int(call_event.get("complaint_history_count", 0) or call_event.get("prior_complaints", 0))
    hour = int(call_event.get("hour_of_day", 12))

    # Keyword frequency counts normalized
    words = re.findall(r"\w+", transcript)
    total_words = max(len(words), 1)

    urgency_hits = sum(1 for kw in URGENCY_KEYWORDS if kw in transcript)
    urgency_score = min(urgency_hits / 3.0, 1.0)

    otp_hits = sum(1 for kw in OTP_CREDENTIAL_KEYWORDS if kw in transcript)
    otp_request_detected = 1 if otp_hits > 0 else 0

    impersonation_hits = sum(1 for kw in IMPERSONATION_KEYWORDS if kw in transcript)
    impersonation_score = min(impersonation_hits / 2.0, 1.0)

    fin_hits = sum(1 for kw in FINANCIAL_DEMAND_KEYWORDS if kw in transcript)
    financial_demand_score = min(fin_hits / 3.0, 1.0)

    # Spoof detection heuristic (e.g., non-standard length or explicitly flagged spoofing header)
    is_spoof = 0
    if call_event.get("is_spoofed_call") or call_event.get("spoof_detected"):
        is_spoof = 1
    elif caller_phone.startswith("+91") and len(re.sub(r"\D", "", caller_phone)) not in (12, 10):
        is_spoof = 1

    # Off-hours indicator (before 8 AM or after 8 PM)
    off_hours = 1 if (hour < 8 or hour >= 20) else 0

    return CallFeatures(
        urgency_score=round(urgency_score, 4),
        otp_request_detected=otp_request_detected,
        impersonation_score=round(impersonation_score, 4),
        financial_demand_score=round(financial_demand_score, 4),
        is_spoof_suspected=is_spoof,
        call_duration_seconds=duration,
        complaint_history_count=complaints,
        off_hours_call=off_hours,
    )
