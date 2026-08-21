"""
Batch retraining job for the transaction anomaly detector.

Spark handles the part that actually needs to scale: reading and
transforming potentially millions of archived JSON event records from
the data lake. Isolation Forest has no native Spark MLlib
implementation, so once Spark reduces that down to one engineered
feature row per transaction (a small matrix, not the raw archive), the
final fit uses scikit-learn -- forcing everything into Spark ML just to
avoid touching sklearn would be worse engineering, not better.

Run against a real MinIO deployment:
    python backend/spark/train_transaction_model_spark.py \\
        --data-path s3a://botocop-datalake/transaction/

Run against local JSON files (e.g. exported from the lake, or testing):
    python backend/spark/train_transaction_model_spark.py \\
        --data-path file:///path/to/transaction_archive/
"""
import argparse
import os

import joblib
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.ensemble import IsolationForest

from backend.src.pipelines.transaction_fraud.features import FEATURE_NAMES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "transaction_iforest.joblib")
STATS_PATH = os.path.join(MODEL_DIR, "transaction_feature_stats.joblib")

HIGH_RISK_COUNTRIES = ["XX", "YY", "ZZ"]
LOW_RISK_COUNTRY_MAP = {"US": 0.05, "CA": 0.05, "UK": 0.1, "DE": 0.1, "FR": 0.1, "AU": 0.05}


def build_spark_session(app_name: str = "botocop-transaction-retrain"):
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("MINIO_ENDPOINT", "http://localhost:9000"))
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY", "minioadmin"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )


def extract_features_with_spark(spark: SparkSession, data_path: str):
    """
    Reads archived event records (written by datalake/writer.py, each a
    JSON object with a nested event_payload) and engineers the same
    feature set the live pipeline's features.py uses -- duplicated as a
    Spark column expression here since it needs to run distributed over
    the archive, not the single dict the live path handles.
    """
    df = spark.read.json(data_path)

    country_risk_expr = F.when(F.col("event_payload.country").isin(HIGH_RISK_COUNTRIES), 0.9)
    for code, risk in LOW_RISK_COUNTRY_MAP.items():
        country_risk_expr = country_risk_expr.when(F.col("event_payload.country") == code, risk)
    country_risk_expr = country_risk_expr.otherwise(0.4)

    features = df.select(
        F.col("event_payload.amount").cast("double").alias("amount"),
        F.hour(F.to_timestamp("event_payload.timestamp")).alias("hour_of_day"),
        F.when(F.col("event_payload.is_new_payee") == True, 1.0).otherwise(0.0).alias("is_new_payee"),  # noqa: E712
        country_risk_expr.alias("country_risk"),
    ).na.fill({"hour_of_day": 12, "amount": 0.0})

    return features.select(*FEATURE_NAMES)


def train(data_path: str, contamination: float = 0.05, model_dir: str = None):
    """model_dir defaults to the real production path; override it in
    tests so a retrain run never clobbers the model other tests depend on."""
    model_dir = model_dir or MODEL_DIR
    model_path = os.path.join(model_dir, "transaction_iforest.joblib")
    stats_path = os.path.join(model_dir, "transaction_feature_stats.joblib")

    spark = build_spark_session()
    try:
        features_df = extract_features_with_spark(spark, data_path)
        n = features_df.count()
        if n == 0:
            raise ValueError(f"No archived records found at {data_path}")

        pdf = features_df.toPandas()
        X = pdf[FEATURE_NAMES].to_numpy()

        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X)

        stats = {
            "mean": X.mean(axis=0).tolist(),
            "std": (X.std(axis=0) + 1e-6).tolist(),
            "feature_names": FEATURE_NAMES,
        }

        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(stats, stats_path)
        print(f"Retrained on {n} archived transactions from {data_path} -> {model_path}")
        return model, stats
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="s3a://bucket/transaction/ or file:///local/path/")
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()
    train(args.data_path, args.contamination)
