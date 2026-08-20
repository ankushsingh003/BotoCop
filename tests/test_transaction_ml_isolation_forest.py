"""
Tests the Isolation Forest transaction-scoring module directly.

Preserved as pipelines/transaction_fraud/nodes_ml_isolation_forest.py after
the transaction pipeline was switched (by explicit choice) back to RAG-based
auditing in nodes.py/workflow.py. Kept here, tested, in case an ML-scoring
path is wanted again later -- it's not currently wired into the orchestrator.
"""
from backend.src.pipelines.transaction_fraud.nodes_ml_isolation_forest import (
    _load_model,
    _most_deviant_feature,
)
from backend.src.pipelines.transaction_fraud.features import transaction_to_features
from backend.src.synthetic.generator import generate_normal_transaction, generate_fraud_transaction


def test_normal_transactions_mostly_not_flagged():
    model, stats = _load_model()
    flagged = 0
    n = 30
    for _ in range(n):
        x = transaction_to_features(generate_normal_transaction())
        if model.predict([x])[0] == -1:
            flagged += 1
    assert flagged / n < 0.2


def test_fraud_transactions_reliably_flagged():
    model, stats = _load_model()
    flagged = 0
    n = 20
    for _ in range(n):
        x = transaction_to_features(generate_fraud_transaction())
        if model.predict([x])[0] == -1:
            flagged += 1
    assert flagged / n > 0.8


def test_deviant_feature_explanation():
    model, stats = _load_model()
    x = transaction_to_features(generate_fraud_transaction())
    feature_name, z = _most_deviant_feature(x, stats)
    assert feature_name in stats["feature_names"]
    assert isinstance(z, float)
