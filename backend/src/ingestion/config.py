import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

TRANSACTION_TOPIC = os.getenv("KAFKA_TRANSACTION_TOPIC", "fraud.transaction.events")
CALL_TOPIC = os.getenv("KAFKA_CALL_TOPIC", "fraud.call.events")
TEXT_TOPIC = os.getenv("KAFKA_TEXT_TOPIC", "fraud.text.events")

RESULTS_TOPIC = os.getenv("KAFKA_RESULTS_TOPIC", "fraud.case.updates")

CONSUMER_GROUP_ID = os.getenv("KAFKA_CONSUMER_GROUP", "fraud-orchestrator")
