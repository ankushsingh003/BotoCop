"""
Kafka consumer -- the real production ingestion path.

Each channel has its own topic; the consumer routes by topic name into
the same orchestrator.handle_event() the WebSocket endpoint uses, so
processing logic is identical no matter how an event arrived. This is
what decouples ingestion from processing: producers publish and move on,
the orchestrator consumes at its own pace, and a slow LLM call or a
processing crash doesn't block or lose the upstream event (Kafka retains
it for replay, unlike an in-memory queue or a blocking HTTP request).
"""
import json
import logging

from backend.src.ingestion.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TRANSACTION_TOPIC,
    CALL_TOPIC,
    TEXT_TOPIC,
    VIDEO_TOPIC,
    RESULTS_TOPIC,
    CONSUMER_GROUP_ID,
)
from backend.src.orchestrator.orchestrator import handle_event

logger = logging.getLogger("kafka-consumer")

TOPIC_TO_CHANNEL = {
    TRANSACTION_TOPIC: "transaction",
    CALL_TOPIC: "call",
    TEXT_TOPIC: "text",
    VIDEO_TOPIC: "video",
}


def process_message(topic: str, payload: dict, producer=None) -> dict:
    """
    Route one message through the orchestrator. Deliberately pulled out of
    the consume loop below so it's testable without a real Kafka broker --
    see tests/test_kafka_consumer.py.
    """
    channel = TOPIC_TO_CHANNEL.get(topic)
    if channel is None:
        raise ValueError(f"No channel mapped for topic '{topic}'")

    logger.info(f"Processing {channel} event from topic '{topic}'")
    result = handle_event(channel, payload)

    if producer is not None:
        producer.send(RESULTS_TOPIC, result)

    return result


def run_consumer(bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS):
    """
    Blocking loop against a real Kafka broker. Requires `kafka-python` and
    a running broker at KAFKA_BOOTSTRAP_SERVERS -- not exercised in tests,
    process_message() above is what's actually tested.
    """
    from kafka import KafkaConsumer, KafkaProducer

    consumer = KafkaConsumer(
        TRANSACTION_TOPIC,
        CALL_TOPIC,
        TEXT_TOPIC,
        VIDEO_TOPIC,
        bootstrap_servers=bootstrap_servers,
        group_id=CONSUMER_GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )

    logger.info(f"Kafka consumer listening on {[TRANSACTION_TOPIC, CALL_TOPIC, TEXT_TOPIC, VIDEO_TOPIC]} @ {bootstrap_servers}")
    for message in consumer:
        try:
            process_message(message.topic, message.value, producer=producer)
        except Exception as e:
            logger.error(f"Failed to process message from {message.topic}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_consumer()
