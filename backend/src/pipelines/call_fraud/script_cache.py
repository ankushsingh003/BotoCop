import re
from typing import Dict, Any, List, Tuple, Optional


class ScamScriptCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ScamScriptCache, cls).__new__(cls)
            cls._instance._init_cache()
        return cls._instance

    def _init_cache(self):
        """Initialize in-memory scam script similarity cache."""
        # List of (canonical_text, set_of_words, violations_list)
        self._cached_scripts: List[Tuple[str, set, List[Dict[str, Any]]]] = []

        # Pre-seed with common canonical vishing scam templates
        self._seed_cache()

    def _canonicalize(self, text: str) -> Tuple[str, set]:
        """Convert transcript into lowercase word set for n-gram / Jaccard similarity."""
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        words = set(w for w in cleaned.split() if len(w) > 2)
        return cleaned, words

    def _seed_cache(self):
        """Seed cache with typical scam templates."""
        seed_templates = [
            (
                "Hello, I am calling from HDFC Bank Support. Your account will be blocked immediately due to missing KYC verification. Share your OTP code right now to unblock.",
                [{
                    "category": "Bank_Impersonation",
                    "description": "Demanded OTP claiming immediate account block for KYC",
                    "severity": "high",
                    "suggestion": "Block caller and alert customer"
                }]
            ),
            (
                "Urgent notification from Mumbai Police Cyber Cell. A legal warrant and arrest order has been issued in your name for money laundering. Transfer funds to safe verification account immediately.",
                [{
                    "category": "Police_Impersonation",
                    "description": "Demanded wire transfer threatening digital arrest and legal warrant",
                    "severity": "critical",
                    "suggestion": "Freeze target account and report to I4C"
                }]
            )
        ]

        for text, violations in seed_templates:
            self.cache_scam_script(text, violations)

    def cache_scam_script(self, transcript: str, violations: List[Dict[str, Any]]):
        """Add an audited scam script to the cache."""
        if not transcript or not violations:
            return
        canon_text, word_set = self._canonicalize(transcript)
        if len(word_set) >= 3:
            self._cached_scripts.append((canon_text, word_set, violations))

    def lookup_cached_script(self, transcript: str, threshold: float = 0.65) -> Tuple[bool, List[Dict[str, Any]], float]:
        """
        Check if incoming transcript is similar to any previously audited scam script.
        Uses Jaccard word set similarity. Returns (hit_found, cached_violations, similarity_score).
        """
        if not transcript:
            return False, [], 0.0

        _, target_words = self._canonicalize(transcript)
        if len(target_words) < 3:
            return False, [], 0.0

        best_sim = 0.0
        best_violations = []

        for _, cached_words, violations in self._cached_scripts:
            intersection = len(target_words.intersection(cached_words))
            union = len(target_words.union(cached_words))
            sim = intersection / max(union, 1)

            if sim > best_sim:
                best_sim = sim
                best_violations = violations

        if best_sim >= threshold:
            return True, best_violations, round(best_sim, 4)

        return False, [], round(best_sim, 4)

    def reset(self):
        """Reset cache (used in unit tests)."""
        self._cached_scripts.clear()
        self._seed_cache()


_script_cache_instance = None

def get_script_cache() -> ScamScriptCache:
    global _script_cache_instance
    if _script_cache_instance is None:
        _script_cache_instance = ScamScriptCache()
    return _script_cache_instance
