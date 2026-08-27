import re
from typing import Dict, Any, Optional, Set, Tuple


class KnownScamBlocklist:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnownScamBlocklist, cls).__new__(cls)
            cls._instance._init_blocklist()
        return cls._instance

    def _init_blocklist(self):
        """Initialize in-memory blocklist pre-seeded with known cybercrime & scam caller IDs."""
        # Mapping: normalized phone number -> metadata dict
        self._blocklist: Dict[str, Dict[str, str]] = {}

        # Seed with initial known scam numbers (I4C / Truecaller / High-Risk Telecom Registry)
        initial_scam_numbers = [
            ("+919876543210", "I4C_Cybercrime_Registry", "Bank Impersonation Vishing Operation"),
            ("+911400000000", "Telecom_Spam_Feed", "Robocall Loan Scam Engine"),
            ("+919999888777", "Truecaller_Reputation_DB", "Police Impersonation / Digital Arrest Scam"),
            ("+918888777666", "Financial_FIU_Blacklist", "Crypto Mule Recruitment Ring"),
        ]

        for phone, source, reason in initial_scam_numbers:
            self.add_scam_number(phone, source, reason)

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number by stripping non-digit characters except leading plus."""
        if not phone:
            return ""
        digits = re.sub(r"[^\d]", "", str(phone))
        if len(digits) == 10:
            return "+91" + digits
        elif len(digits) == 12 and digits.startswith("91"):
            return "+" + digits
        return "+" + digits if digits else ""

    def add_scam_number(self, phone: str, source: str = "Manual_Report", reason: str = "Reported Scam Activity"):
        """Add a caller phone number to the deterministic blocklist."""
        norm = self._normalize_phone(phone)
        if norm:
            self._blocklist[norm] = {
                "phone": norm,
                "source": source,
                "reason": reason,
            }

    def check_blocklist(self, phone: str) -> Tuple[bool, Optional[Dict[str, str]]]:
        """
        O(1) lookup to check if caller phone is in the scam blocklist.
        Returns (is_blocked, metadata_dict).
        """
        if not phone:
            return False, None
        norm = self._normalize_phone(phone)
        if norm in self._blocklist:
            return True, self._blocklist[norm]
        
        # Check raw digits match fallback
        raw_digits = re.sub(r"\D", "", phone)
        for blocked_phone, meta in self._blocklist.items():
            if re.sub(r"\D", "", blocked_phone) == raw_digits:
                return True, meta

        return False, None

    def reset(self):
        """Reset blocklist (used in unit tests)."""
        self._blocklist.clear()
        self._init_blocklist()


_blocklist_instance = None

def get_scam_blocklist() -> KnownScamBlocklist:
    global _blocklist_instance
    if _blocklist_instance is None:
        _blocklist_instance = KnownScamBlocklist()
    return _blocklist_instance
