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


# Mock identity resolution table to map contact info to account IDs.
# In a real system, this would be a lookup against a customer master database.
MOCK_IDENTITY_MAP = {
    "rohan@example.com": "ACC-7781",
    "+919876543210": "ACC-7781"
}

def resolve_entity_id(channel: str, event_payload: dict) -> Optional[str]:
    # 1. Try to find an explicit account ID first
    explicit_id = (
        event_payload.get("account_id") 
        or event_payload.get("linked_account_id")
        or event_payload.get("customer_id") 
        or event_payload.get("user_id")
    )
    if explicit_id:
        return explicit_id

    # 2. If no explicit account ID, fall back to channel-specific contact info 
    # and map it to an account ID using our mock identity graph.
    contact_info = None
    if channel == "call":
        # The victim's phone number (not the caller/attacker)
        contact_info = event_payload.get("phone_number") or event_payload.get("recipient_phone")
    elif channel == "text":
        # The victim's email (not the sender/attacker)
        contact_info = event_payload.get("recipient_email")

    if contact_info:
        # Try to map the contact info to an account ID
        mapped_account = MOCK_IDENTITY_MAP.get(contact_info)
        if mapped_account:
            logger.info(f"Identity resolution: mapped {contact_info} -> {mapped_account}")
            return mapped_account
        
        # If we can't map it, use the contact info itself as the entity ID as a last resort
        return contact_info

    logger.warning(f"Unknown channel '{channel}' or missing keys; cannot resolve entity_id")
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
