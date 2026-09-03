import time
from collections import defaultdict, deque
from typing import Dict, Any, List


class CallVelocityAnalyzer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CallVelocityAnalyzer, cls).__new__(cls)
            cls._instance._init_tracker()
        return cls._instance

    def _init_tracker(self):
        """Initialize in-memory sliding window queue and target account mapping."""
        # Mapping: caller_phone -> deque of (timestamp, target_customer_id)
        self._caller_history: Dict[str, deque] = defaultdict(deque)
        # Mapping: caller_phone -> set of all lifetime target customer_ids
        self._caller_lifetime_targets: Dict[str, set] = defaultdict(set)
        # Window size: 3600 seconds (1 hour)
        self.window_seconds = 3600

    def record_and_analyze(self, caller_phone: str, target_customer_id: str) -> Dict[str, Any]:
        """
        Record an incoming call event timestamp and calculate velocity & fan-out metrics.
        
        Returns:
            - call_velocity_1h (int): Number of calls in past 1h
            - distinct_targets_1h (int): Number of unique target accounts in past 1h
            - fanout_ratio_1h (float): distinct_targets / call_velocity (0.0 to 1.0)
            - cross_account_target_count (int): Total lifetime unique accounts targeted
        """
        now = time.time()

        if not caller_phone:
            return {
                "call_velocity_1h": 1,
                "distinct_targets_1h": 1,
                "fanout_ratio_1h": 1.0,
                "cross_account_target_count": 1,
            }

        target_id = target_customer_id or "UNKNOWN"
        history = self._caller_history[caller_phone]

        # Record current event
        history.append((now, target_id))
        self._caller_lifetime_targets[caller_phone].add(target_id)

        # Evict events older than window_seconds (1 hour)
        cutoff = now - self.window_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

        call_velocity_1h = len(history)
        unique_targets_1h = {item[1] for item in history if item[1] != "UNKNOWN"}
        distinct_targets_1h = max(len(unique_targets_1h), 1)

        fanout_ratio_1h = round(distinct_targets_1h / max(call_velocity_1h, 1), 4)
        cross_account_target_count = len(self._caller_lifetime_targets[caller_phone])

        return {
            "call_velocity_1h": call_velocity_1h,
            "distinct_targets_1h": distinct_targets_1h,
            "fanout_ratio_1h": fanout_ratio_1h,
            "cross_account_target_count": cross_account_target_count,
        }

    def reset(self):
        """Reset velocity data (used in tests)."""
        self._caller_history.clear()
        self._caller_lifetime_targets.clear()


_velocity_analyzer = None

def get_velocity_analyzer() -> CallVelocityAnalyzer:
    global _velocity_analyzer
    if _velocity_analyzer is None:
        _velocity_analyzer = CallVelocityAnalyzer()
    return _velocity_analyzer
