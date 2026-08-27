"""
Prometheus metrics for the fraud detection pipeline.

Exposed at /metrics on the FastAPI app. Prometheus scrapes that endpoint
on an interval (see docker/prometheus.yml) and stores the time series;
Grafana then queries Prometheus for dashboards -- Prometheus pulls from
the app, Grafana pulls from Prometheus, nothing pushes anywhere. This
module only defines and updates the metrics; it has no dependency on
either Prometheus or Grafana directly.

Instrumented once at orchestrator.handle_event() rather than separately
in the WebSocket route and the Kafka consumer -- every event flows
through handle_event regardless of how it arrived, so one instrumentation
point covers both ingestion paths.
"""
from prometheus_client import Counter, Histogram, Gauge

EVENTS_PROCESSED = Counter(
    "botocop_events_processed_total",
    "Total events processed, by channel and final pipeline status",
    ["channel", "final_status"],
)

VIOLATIONS_DETECTED = Counter(
    "botocop_violations_detected_total",
    "Total violations found, by channel and severity",
    ["channel", "severity"],
)

PIPELINE_DURATION_SECONDS = Histogram(
    "botocop_pipeline_duration_seconds",
    "Time spent in the specialist pipeline including its eval/retry loop, by channel",
    ["channel"],
)

EVAL_RETRIES = Counter(
    "botocop_eval_retries_total",
    "Total eval-loop retries triggered (not-confident-yet), by channel",
    ["channel"],
)

CASE_STATUS_TRANSITIONS = Counter(
    "botocop_case_status_transitions_total",
    "Total case status transitions decided by the cross-channel judge",
    ["status"],
)

CASE_RISK_SCORE = Histogram(
    "botocop_case_risk_score",
    "Distribution of computed cross-channel case risk scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

OPEN_CASES = Gauge(
    "botocop_open_cases",
    "Current number of open (non-terminal) cases",
)

CALL_ML_FRAUD_PROBABILITY = Histogram(
    "botocop_call_ml_fraud_probability",
    "Distribution of Call Fraud ML Model predicted probabilities",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)

CALL_ML_RISK_LEVEL = Counter(
    "botocop_call_ml_risk_level_total",
    "Total call fraud risk levels assigned by ML model",
    ["risk_level"],
)



def _init_open_cases_gauge():
    """
    set_function's callback runs at Prometheus scrape/collection time,
    not once at import -- so this stays accurate without needing to
    increment/decrement it inline everywhere a case's status changes and
    risking it drifting out of sync with the DB.
    """
    from backend.src.case.store import count_open_cases
    OPEN_CASES.set_function(count_open_cases)


_init_open_cases_gauge()
