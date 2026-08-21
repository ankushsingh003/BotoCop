"""
Trains the transaction anomaly detector on synthetic "normal" data.

Unsupervised (Isolation Forest) on purpose: we don't have labeled fraud
data, and in a real deployment fraud patterns drift, so anomaly detection
against a model of "normal" generalizes better than a classifier trained
on a fixed, quickly-stale set of known fraud examples.

Run: python backend/scripts/train_transaction_model.py
"""
import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from backend.src.synthetic.generator import generate_training_dataset
from backend.src.pipelines.transaction_fraud.features import transaction_to_features, FEATURE_NAMES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "transaction_iforest.joblib")
STATS_PATH = os.path.join(MODEL_DIR, "transaction_feature_stats.joblib")


def train(n_samples: int = 500, contamination: float = 0.05):
    dataset = generate_training_dataset(n_samples)
    X = np.array([transaction_to_features(t) for t in dataset])

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)

    stats = {
        "mean": X.mean(axis=0).tolist(),
        "std": (X.std(axis=0) + 1e-6).tolist(),
        "feature_names": FEATURE_NAMES,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(stats, STATS_PATH)
    print(f"Trained on {n_samples} synthetic normal transactions -> {MODEL_PATH}")
    return model, stats


if __name__ == "__main__":
    train()
