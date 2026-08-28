import os
import time
import logging
from collections import defaultdict, deque
from typing import Dict, Any, List

logger = logging.getLogger("call-velocity-analyzer")


class CallVelocityAnalyzer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CallVelocityAnalyzer, cls).__new__(cls)
            cls._instance._init_tracker()
        return cls._instance

    def _init_tracker(self):
        """Initialize sliding window tracker with Redis distributed support and DB fallback."""
        self._caller_history: Dict[str, deque] = defaultdict(deque)
        self._caller_lifetime_targets: Dict[str, set] = defaultdict(set)
        self.window_seconds = 3600

        # Attempt connecting to Redis for distributed state
        self.redis_client = None
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis
            client = redis.Redis.from_url(redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)
            client.ping()
            self.redis_client = client
            logger.info("CallVelocityAnalyzer: Connected to distributed Redis store.")
        except Exception:
            logger.info("CallVelocityAnalyzer: Redis unavailable; using DB fallback + local state.")

    def record_and_analyze(self, caller_phone: str, target_customer_id: str) -> Dict[str, Any]:
        """
        Record an incoming call event timestamp and calculate velocity & fan-out metrics.
        Supported backends: Distributed Redis ZSET -> Persistent SQL DB Fallback -> L1 Cache.
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

        # 1. Distributed Redis Implementation
        if self.redis_client:
            try:
                hist_key = f"bocop:vel:hist:{caller_phone}"
                target_key = f"bocop:vel:targets:{caller_phone}"

                # Add event score to ZSET
                self.redis_client.zadd(hist_key, {f"{now}:{target_id}": now})
                self.redis_client.sadd(target_key, target_id)
                self.redis_client.expire(hist_key, self.window_seconds + 60)

                # Remove events older than 1h
                cutoff = now - self.window_seconds
                self.redis_client.zremrangebyscore(hist_key, 0, cutoff)

                # Query global counts across all server instances
                call_velocity_1h = self.redis_client.zcard(hist_key)
                all_events = self.redis_client.zrange(hist_key, 0, -1)
                
                unique_targets = set()
                for item in all_events:
                    item_str = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    if ":" in item_str:
                        _, t_id = item_str.split(":", 1)
                        if t_id != "UNKNOWN":
                            unique_targets.add(t_id)

                distinct_targets_1h = max(len(unique_targets), 1)
                fanout_ratio_1h = round(distinct_targets_1h / max(call_velocity_1h, 1), 4)
                cross_account_target_count = self.redis_client.scard(target_key)

                return {
                    "call_velocity_1h": call_velocity_1h,
                    "distinct_targets_1h": distinct_targets_1h,
                    "fanout_ratio_1h": fanout_ratio_1h,
                    "cross_account_target_count": cross_account_target_count,
                }
            except Exception as e:
                logger.warning(f"Redis velocity query failed ({e}); falling back to DB/local storage.")

        # 2. Database Fallback (SQL CaseStore query for multi-instance sync)
        db_events_count = 0
        db_unique_targets = set()
        try:
            from backend.src.case.models import Case, CaseEvent
            from backend.src.case.db import get_session
            from datetime import datetime, timedelta, timezone
            
            session = get_session()
            cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds)
            
            recent_cases = session.query(Case).filter(Case.entity_id == caller_phone).all()
            for c in recent_cases:
                events = session.query(CaseEvent).filter(
                    CaseEvent.case_id == c.case_id,
                    CaseEvent.created_at >= cutoff_dt
                ).all()
                for ev in events:
                    db_events_count += 1
                    t_id = (ev.pipeline_result or {}).get("target_customer_id", "UNKNOWN")
                    if t_id != "UNKNOWN":
                        db_unique_targets.add(t_id)
            session.close()
        except Exception:
            pass

        # 3. Local L1 Cache recording
        history = self._caller_history[caller_phone]
        history.append((now, target_id))
        self._caller_lifetime_targets[caller_phone].add(target_id)

        cutoff = now - self.window_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

        local_velocity = len(history)
        local_targets = {item[1] for item in history if item[1] != "UNKNOWN"}

        # Combine local L1 state with DB historical events
        call_velocity_1h = max(local_velocity, db_events_count + 1)
        combined_unique_targets = local_targets.union(db_unique_targets)
        distinct_targets_1h = max(len(combined_unique_targets), 1)

        fanout_ratio_1h = round(distinct_targets_1h / max(call_velocity_1h, 1), 4)
        cross_account_target_count = max(len(self._caller_lifetime_targets[caller_phone]), len(combined_unique_targets))

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

