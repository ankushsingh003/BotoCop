from datetime import datetime, timezone, timedelta
from typing import Literal, Optional
from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field, constr, field_validator, AwareDatetime

def now():
    return datetime.now(timezone.utc)

class TransactionV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: Literal[1]
    event_id: str
    event_time: AwareDatetime
    account_id: str = Field(min_length=1)
    amount: Decimal = Field(gt=0, max_digits=14, decimal_places=2)
    currency: constr(pattern=r"^[A-Z]{3}$") = "USD"
    payee_id: Optional[str] = None
    is_new_payee: bool = False
    ip_address: Optional[str] = None
    merchant: Optional[str] = None

    @field_validator("event_time")
    def not_future(cls, v):
        if v > now() + timedelta(minutes=5): raise ValueError("future timestamp")
        if v < now() - timedelta(days=14): raise ValueError("too old timestamp")
        return v

class CallV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: Literal[1] = 1 # Allow default for backward compat during migration
    event_id: str
    event_time: Optional[AwareDatetime] = None
    caller_phone: str
    linked_account_id: Optional[str] = None
    phone_number: Optional[str] = None
    recipient_phone: Optional[str] = None
    transcript: str
    duration_seconds: int = Field(ge=0)
    stir_shaken_attestation: Optional[str] = None
    line_type: Optional[str] = None
    hour_of_day: Optional[int] = None

    @field_validator("event_time", mode="before")
    def populate_time_if_missing(cls, v):
        if v is None: return now()
        return v

    @field_validator("event_time")
    def validate_time(cls, v):
        if v > now() + timedelta(minutes=5): raise ValueError("future timestamp")
        if v < now() - timedelta(days=14): raise ValueError("too old timestamp")
        return v

class TextV1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: Literal[1] = 1
    event_id: str
    event_time: Optional[AwareDatetime] = None
    sender_email: Optional[str] = None
    recipient_email: Optional[str] = None
    body: str

    @field_validator("event_time", mode="before")
    def populate_time_if_missing(cls, v):
        if v is None: return now()
        return v

def validate_payload(channel: str, payload: dict) -> dict:
    """Validates an incoming payload against the defined schema contract for the channel."""
    # Ensure event_id exists before pydantic check for easier migration, or let pydantic fail.
    # Actually, orchestrator adds event_id if missing, so let's rely on orchestrator passing it.
    
    if channel == "transaction":
        return TransactionV1.model_validate(payload).model_dump()
    elif channel == "call":
        return CallV1.model_validate(payload).model_dump()
    elif channel == "text":
        return TextV1.model_validate(payload).model_dump()
    
    raise ValueError(f"Unknown channel schema: {channel}")
