import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# One topic per channel -- lets each upstream producer (whatever system
# emits transactions vs. calls) publish independently without agreeing
# on a shared envelope schema beyond "this topic is transaction events".
TRANSACTION_TOPIC = os.getenv("KAFKA_TRANSACTION_TOPIC", "fraud.transaction.events")
CALL_TOPIC = os.getenv("KAFKA_CALL_TOPIC", "fraud.call.events")
TEXT_TOPIC = os.getenv("KAFKA_TEXT_TOPIC", "fraud.text.events")

# Where processed results get published, so downstream systems (a
# dashboard, an alerting service) can consume outcomes without querying
# the case DB directly.
RESULTS_TOPIC = os.getenv("KAFKA_RESULTS_TOPIC", "fraud.case.updates")

CONSUMER_GROUP_ID = os.getenv("KAFKA_CONSUMER_GROUP", "fraud-orchestrator")
