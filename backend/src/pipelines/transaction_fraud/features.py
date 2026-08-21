"""
Feature extraction for transaction fraud detection.

Deliberately a small, explainable feature set (not learned embeddings)
so the anomaly detector's output can be traced back to a specific
deviant feature -- that's what makes the violation description
meaningful instead of a black-box "0.83 anomaly score".
"""
from datetime import datetime

import numpy as np

COUNTRY_RISK = {"US": 0.05, "CA": 0.05, "UK": 0.1, "DE": 0.1, "FR": 0.1, "AU": 0.05}
DEFAULT_COUNTRY_RISK = 0.4
HIGH_RISK_COUNTRY_CODES = {"XX", "YY", "ZZ"}  # synthetic placeholders, see generator.py

FEATURE_NAMES = ["amount", "hour_of_day", "is_new_payee", "country_risk"]


def country_risk_score(country: str) -> float:
    if country in HIGH_RISK_COUNTRY_CODES:
        return 0.9
    return COUNTRY_RISK.get(country, DEFAULT_COUNTRY_RISK)


def extract_hour(timestamp_str: str) -> int:
    try:
        return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")).hour
    except Exception:
        return 12  # assume midday if missing/unparseable, avoids spurious flags


def transaction_to_features(transaction: dict) -> np.ndarray:
    amount = float(transaction.get("amount", 0))
    hour = extract_hour(transaction.get("timestamp", ""))
    is_new_payee = 1.0 if transaction.get("is_new_payee") else 0.0
    country_risk = country_risk_score(transaction.get("country", ""))
    return np.array([amount, hour, is_new_payee, country_risk], dtype=float)
