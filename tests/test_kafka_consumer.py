"""
Tests the message-routing logic in isolation from the actual Kafka
broker connection. run_consumer() is a thin loop around process_message()
that requires a real broker (kafka-python + a running Kafka instance) --
not exercised here, since no broker is available in this sandbox. This
proves the routing/processing contract is correct; the broker plumbing
itself is standard kafka-python usage with nothing project-specific to
test.
"""
import pytest
from unittest.mock import MagicMock

from backend.src.ingestion import kafka_consumer
from backend.src.ingestion.config import TRANSACTION_TOPIC, CALL_TOPIC, RESULTS_TOPIC


def test_process_message_routes_transaction_topic(monkeypatch):
    captured = {}

    def fake_handle_event(channel, payload):
        captured["channel"] = channel
        captured["payload"] = payload
        return {"case_id": "abc-123", "channel": channel}

    monkeypatch.setattr(kafka_consumer, "handle_event", fake_handle_event)

    result = kafka_consumer.process_message(TRANSACTION_TOPIC, {"account_id": "acct_1", "amount": 100})

    assert captured["channel"] == "transaction"
    assert result["channel"] == "transaction"


def test_process_message_routes_call_topic(monkeypatch):
    captured = {}

    def fake_handle_event(channel, payload):
        captured["channel"] = channel
        return {"case_id": "abc-456", "channel": channel}

    monkeypatch.setattr(kafka_consumer, "handle_event", fake_handle_event)

    kafka_consumer.process_message(CALL_TOPIC, {"linked_account_id": "acct_1", "transcript": "hi"})

    assert captured["channel"] == "call"


def test_process_message_unknown_topic_raises():
    with pytest.raises(ValueError):
        kafka_consumer.process_message("unmapped.topic", {})


def test_process_message_publishes_result_to_producer(monkeypatch):
    monkeypatch.setattr(kafka_consumer, "handle_event", lambda c, p: {"case_id": "abc-789"})
    fake_producer = MagicMock()

    kafka_consumer.process_message(
        CALL_TOPIC, {"linked_account_id": "acct_1", "transcript": "hi"}, producer=fake_producer
    )

    fake_producer.send.assert_called_once_with(RESULTS_TOPIC, {"case_id": "abc-789"})
