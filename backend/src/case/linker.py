"""
Entity linking: maps an incoming channel event to a case.

Scope note (intentional, not an oversight): this links strictly by an
explicit shared identifier already present in the event payload
(account_id for transactions, a linked account id or phone number for
calls). It does NOT attempt fuzzy cross-channel identity resolution
(e.g. inferring that an email and a phone number belong to the same
person with no shared key) -- that's a hard problem on its own and out
of scope here. The assumption is that an account-to-phone-number
mapping already exists as account metadata upstream.
"""
import logging
from typing import Optional

from backend.src.case.store import get_open_case_for_entity, create_case
from backend.src.case.models import Case

logger = logging.getLogger("case-linker")


def resolve_entity_id(channel: str, event_payload: dict) -> Optional[str]:
    if channel == "transaction":
        return event_payload.get("account_id") or event_payload.get("customer_id") or event_payload.get("user_id")
    if channel == "call":
        return (
            event_payload.get("linked_account_id")
            or event_payload.get("account_id")
            or event_payload.get("phone_number")
            or event_payload.get("caller_phone")
            or event_payload.get("customer_id")
            or event_payload.get("user_id")
        )
    if channel == "text":
        return (
            event_payload.get("linked_account_id")
            or event_payload.get("account_id")
            or event_payload.get("sender_email")
            or event_payload.get("customer_id")
            or event_payload.get("user_id")
        )
    logger.warning(f"Unknown channel '{channel}'; cannot resolve entity_id")
    return None




def link_event_to_case(channel: str, event_payload: dict) -> Case:
    """Find the open case for this event's entity, or open a new one."""
    entity_id = resolve_entity_id(channel, event_payload)
    if not entity_id:
        raise ValueError(f"Could not resolve entity_id for {channel} event: {event_payload}")

    case = get_open_case_for_entity(entity_id)
    if case is not None:
        logger.info(f"Linked {channel} event to existing case {case.case_id} (entity={entity_id})")
        return case

    return create_case(entity_id)
