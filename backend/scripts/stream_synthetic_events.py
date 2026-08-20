"""
Publishes a stream of synthetic transaction/call events onto Kafka --
useful for demoing the live pipeline against a real broker.

Requires a running Kafka broker at KAFKA_BOOTSTRAP_SERVERS
(default localhost:9092). Run the consumer in a separate process first:
    python -m backend.src.ingestion.kafka_consumer
then:
    python backend/scripts/stream_synthetic_events.py
"""
import random
import time

from backend.src.ingestion.kafka_producer import get_producer, publish_event
from backend.src.synthetic.generator import (
    generate_normal_transaction,
    generate_normal_call,
    generate_correlated_fraud_case,
)


def stream(n_events: int = 50, delay_seconds: float = 1.0, fraud_case_rate: float = 0.1):
    producer = get_producer()
    for _ in range(n_events):
        if random.random() < fraud_case_rate:
            call, txn = generate_correlated_fraud_case()
            publish_event(producer, "call", call)
            time.sleep(delay_seconds)
            publish_event(producer, "transaction", txn)
        elif random.random() < 0.5:
            publish_event(producer, "transaction", generate_normal_transaction())
        else:
            publish_event(producer, "call", generate_normal_call())
        time.sleep(delay_seconds)
    producer.flush()
    print(f"Streamed {n_events} events (~{fraud_case_rate:.0%} correlated fraud cases).")


if __name__ == "__main__":
    stream()
