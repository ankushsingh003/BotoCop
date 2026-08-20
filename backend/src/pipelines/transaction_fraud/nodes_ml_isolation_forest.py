import logging
import os
from typing import Dict, Any

import joblib
import numpy as np

from backend.src.pipelines.transaction_fraud.state import TransactionFraudState
from backend.src.pipelines.transaction_fraud.features import transaction_to_features

logger = logging.getLogger("transaction-fraud")

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "transaction_iforest.joblib")
_STATS_PATH = os.path.join(_MODEL_DIR, "transaction_feature_stats.joblib")

_model = None
_stats = None


def _load_model():
    global _model, _stats
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model at {_MODEL_PATH}. "
                f"Run `python backend/scripts/train_transaction_model.py` first."
            )
        _model = joblib.load(_MODEL_PATH)
        _stats = joblib.load(_STATS_PATH)
    return _model, _stats


def _most_deviant_feature(x: np.ndarray, stats: dict):
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    z_scores = (x - mean) / std
    idx = int(np.argmax(np.abs(z_scores)))
    return stats["feature_names"][idx], float(z_scores[idx])


def score_transaction_node(state: TransactionFraudState) -> Dict[str, Any]:
    """
    Unsupervised anomaly detection over transaction features. Structured
    numeric fraud signals (amount, timing, payee novelty, country risk)
    belong to an ML model, not an LLM reading a compliance PDF -- this
    replaces the earlier RAG-based approach entirely.
    """
    transaction = state.get("transaction") or {}
    try:
        model, stats = _load_model()
    except FileNotFoundError as e:
        logger.error(str(e))
        return {"error": [str(e)], "violations": [], "final_status": "failed"}

    x = transaction_to_features(transaction)
    anomaly_score = float(model.decision_function([x])[0])  # higher = more normal
    is_outlier = bool(model.predict([x])[0] == -1)

    violations = []
    if is_outlier:
        feature_name, z = _most_deviant_feature(x, stats)
        severity = "high" if anomaly_score < -0.15 else "medium"
        violations.append({
            "category": "Anomalous_Transaction",
            "description": (
                f"Transaction flagged as anomalous (isolation-forest score={anomaly_score:.3f}); "
                f"most deviant feature: {feature_name} (z={z:.2f})"
            ),
            "severity": severity,
            "suggestion": "Hold for manual review before settlement.",
        })

    return {
        "violations": violations,
        "final_status": "warning" if violations else "success",
    }
