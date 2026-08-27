import time
import logging
from typing import Dict, Any, List, Optional
from backend.src.pipelines.call_fraud.blocklist import get_scam_blocklist

logger = logging.getLogger("hitl-queue")


class HITLReviewQueue:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HITLReviewQueue, cls).__new__(cls)
            cls._instance._init_queue()
        return cls._instance

    def _init_queue(self):
        """Initialize in-memory queue for Human-In-The-Loop analyst review."""
        # Mapping: case_id -> review case payload dict
        self._pending_reviews: Dict[str, Dict[str, Any]] = {}
        self._resolved_reviews: Dict[str, Dict[str, Any]] = {}

    def enqueue_case(
        self,
        case_id: str,
        call_data: Dict[str, Any],
        ml_score: Dict[str, Any],
        violations: List[Dict[str, Any]],
        identity_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Enqueue a high-risk or flagged call case for human analyst review.
        """
        caller_phone = call_data.get("caller_phone") or call_data.get("phone_number") or ""
        audio_url = call_data.get("audio_url") or call_data.get("recording_path") or "s3://botocop-voice-vault/recordings/sample.wav"

        review_item = {
            "case_id": case_id,
            "caller_phone": caller_phone,
            "target_account_id": call_data.get("linked_account_id") or call_data.get("customer_id") or "UNKNOWN",
            "audio_url": audio_url,
            "transcript": call_data.get("transcript", ""),
            "ml_score": ml_score,
            "violations": violations,
            "identity_graph": identity_graph or {},
            "review_status": "PENDING_HUMAN_REVIEW",
            "enqueued_at": time.time(),
            "analyst_id": None,
            "analyst_notes": None,
            "resolution_action": None,
        }

        self._pending_reviews[case_id] = review_item
        logger.info(f"Enqueued Case {case_id} for HITL Analyst Review (Caller: {caller_phone}, Risk: {ml_score.get('risk_level')})")
        return review_item

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Return all cases awaiting human analyst verification."""
        return list(self._pending_reviews.values())

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific review case by ID."""
        return self._pending_reviews.get(case_id) or self._resolved_reviews.get(case_id)

    def resolve_review(
        self,
        case_id: str,
        analyst_id: str,
        decision: str,  # "CONFIRM_FRAUD" or "OVERRIDE_FALSE_POSITIVE"
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Analyst submits review decision:
        - CONFIRM_FRAUD: Confirms scam. Automatically adds caller ID to Layer 4 Blocklist!
        - OVERRIDE_FALSE_POSITIVE: Dismisses flag and unlocks account.
        """
        case = self._pending_reviews.pop(case_id, None)
        if not case:
            if case_id in self._resolved_reviews:
                return self._resolved_reviews[case_id]
            raise ValueError(f"Case {case_id} not found in pending review queue.")

        decision_upper = decision.upper()
        if decision_upper == "CONFIRM_FRAUD":
            case["review_status"] = "CONFIRMED_FRAUD"
            case["resolution_action"] = "ENFORCE_BLOCK_AND_ADD_TO_BLACK_LIST"
            # Automatically feed confirmed bad caller into Layer 4 Blocklist!
            caller_phone = case.get("caller_phone")
            if caller_phone:
                get_scam_blocklist().add_scam_number(
                    phone=caller_phone,
                    source=f"HITL_Analyst_{analyst_id}",
                    reason=notes or f"Human Analyst Verified Vishing Scam (Case: {case_id})"
                )
                logger.info(f"HITL CONFIRMED: Caller {caller_phone} added to Layer 4 Blocklist by Analyst {analyst_id}")
        elif decision_upper in ["OVERRIDE_FALSE_POSITIVE", "DISMISS"]:
            case["review_status"] = "OVERRIDDEN_FALSE_POSITIVE"
            case["resolution_action"] = "DISMISS_FLAG_AND_UNFREEZE"
            logger.info(f"HITL OVERRIDE: Case {case_id} dismissed as False Positive by Analyst {analyst_id}")
        else:
            raise ValueError(f"Invalid decision '{decision}'. Use 'CONFIRM_FRAUD' or 'OVERRIDE_FALSE_POSITIVE'.")

        case["analyst_id"] = analyst_id
        case["analyst_notes"] = notes
        case["resolved_at"] = time.time()

        self._resolved_reviews[case_id] = case
        return case

    def reset(self):
        """Reset queue data (used in unit tests)."""
        self._pending_reviews.clear()
        self._resolved_reviews.clear()


_hitl_queue_instance = None

def get_hitl_queue() -> HITLReviewQueue:
    global _hitl_queue_instance
    if _hitl_queue_instance is None:
        _hitl_queue_instance = HITLReviewQueue()
    return _hitl_queue_instance
