import os
import time
import json
import logging
import hashlib
import random
from typing import Dict, Any, Type
from threading import Semaphore, Lock
from pydantic import BaseModel, ValidationError

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("llm-gateway")

class LLMUnavailable(Exception):
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
        self._lock = Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF-OPEN"
                    return True
                return False
            return True

    def success(self):
        with self._lock:
            self.failures = 0
            self.state = "CLOSED"

    def failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"

class Cache:
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self._cache = {}
        self._lock = Lock()

    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                val, expires_at = self._cache[key]
                if time.time() < expires_at:
                    return val
                del self._cache[key]
            return None

    def set(self, key: str, value: Any):
        with self._lock:
            self._cache[key] = (value, time.time() + self.ttl)

class LLMGateway:
    def __init__(self):
        # Models configuration (primary and fallback)
        primary_model = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        secondary_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-1.5-pro")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        
        self.models = {
            "event_audit": [primary_model, secondary_model],
            "case_judge": [primary_model, secondary_model],
        }
        self.api_key = api_key
        
        self.breakers = {
            "event_audit": CircuitBreaker(),
            "case_judge": CircuitBreaker()
        }
        
        self.limiters = {
            "event_audit": Semaphore(5), # 5 concurrent calls
            "case_judge": Semaphore(2)   # 2 concurrent calls (higher priority/slower)
        }
        
        self.cache = Cache(ttl=300)
        
    def _extract_json(self, text: str) -> dict:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        json_str = text[start_idx:end_idx + 1] if start_idx != -1 and end_idx != -1 else text
        return json.loads(json_str)

    def invoke(self, task: str, system_prompt: str, user_prompt: str, schema: Type[BaseModel]) -> BaseModel:
        if not self.breakers[task].allow():
            raise LLMUnavailable("circuit open")
            
        # Check cache
        cache_key = hashlib.sha256(f"{task}:{system_prompt}:{user_prompt}".encode()).hexdigest()
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return schema.model_validate(cached_result)

        for model_name in self.models[task]:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=self.api_key, request_timeout=20.0)
            
            for attempt in range(3):
                with self.limiters[task]:
                    try:
                        start = time.perf_counter()
                        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
                        latency = time.perf_counter() - start
                        
                        data = self._extract_json(response.content)
                        validated = schema.model_validate(data)
                        
                        self.breakers[task].success()
                        self.cache.set(cache_key, data)
                        
                        # In production, emit metrics: tokens, latency, etc.
                        logger.info(f"LLM Success (task={task}, model={model_name}, attempt={attempt}, latency={latency:.2f}s)")
                        return validated
                        
                    except ValidationError as e:
                        logger.warning(f"LLM Schema validation failed: {e}")
                        break # Try next model, this model is returning bad format
                    except Exception as e:
                        logger.error(f"LLM Error on {model_name} attempt {attempt}: {e}")
                        time.sleep((2 ** attempt) + random.uniform(0, 1)) # Exponential backoff + jitter
                        
        self.breakers[task].failure()
        raise LLMUnavailable(task)

gateway = LLMGateway()

def get_gateway() -> LLMGateway:
    return gateway
