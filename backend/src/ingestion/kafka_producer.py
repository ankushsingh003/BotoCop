"""
Thin producer wrapper for publishing events onto the ingestion topics.
Used by upstream systems (or, for demo/testing, the synthetic stream
script) to simulate a live transaction/call feed.
"""
import json
import logging

from backend.src.ingestion.config import KAFKA_BOOTSTRAP_SERVERS, TRANSACTION_TOPIC, CALL_TOPIC, TEXT_TOPIC, VIDEO_TOPIC

logger = logging.getLogger("kafka-producer")

CHANNEL_TO_TOPIC = {
    "transaction": TRANSACTION_TOPIC,
    "call": CALL_TOPIC,
    "text": TEXT_TOPIC,
    "video": VIDEO_TOPIC,
}


def get_producer(bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS):
    from kafka import KafkaProducer
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )


def publish_event(producer, channel: str, payload: dict):
    topic = CHANNEL_TO_TOPIC.get(channel)
    if topic is None:
        raise ValueError(f"Unknown channel '{channel}'")
    producer.send(topic, payload)
    logger.info(f"Published {channel} event to {topic}")
