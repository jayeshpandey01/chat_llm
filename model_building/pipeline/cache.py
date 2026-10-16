"""
High-Speed Canonical Symptom Cache Layer.
=========================================
Implements dual-tier deterministic caching (Raw Text Key + Canonical Symptom Key)
and an in-memory LRU cache delivering sub-1ms responses for repeated queries.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, List, Optional

from model_building.pipeline.schemas import DiagnosticReportSchema


class CanonicalSymptomCache:
    """Thread-safe multi-tier LRU cache for clinical symptom predictions."""

    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._cache: OrderedDict[str, tuple[DiagnosticReportSchema, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _categorize_age(age: Optional[float]) -> str:
        if age is None:
            return "unspecified"
        if age < 18.0:
            return "pediatric"
        if age < 65.0:
            return "adult"
        return "geriatric"

    @staticmethod
    def _normalize_token(token: str) -> str:
        return token.strip().lower().replace(" ", "_")

    def make_raw_key(
        self,
        query_text: Optional[str] = None,
        selected_symptoms: Optional[List[str]] = None,
        age: Optional[float] = None,
        biological_sex: Optional[str] = None,
    ) -> Optional[str]:
        """Constructs a fast raw-input SHA-256 hash key."""
        q_norm = (query_text or "").strip().lower()
        sel_norm = sorted(set(self._normalize_token(s) for s in (selected_symptoms or []) if s.strip()))
        if not q_norm and not sel_norm:
            return None

        age_bucket = self._categorize_age(age)
        sex_bucket = (biological_sex or "unspecified").strip().lower()

        raw_repr = f"raw_q:{q_norm}|sel:{','.join(sel_norm)}|age:{age_bucket}|sex:{sex_bucket}"
        return hashlib.sha256(raw_repr.encode("utf-8")).hexdigest()

    def make_canonical_key(
        self,
        present_symptoms: List[str],
        denied_symptoms: Optional[List[str]] = None,
        age: Optional[float] = None,
        biological_sex: Optional[str] = None,
    ) -> str:
        """Constructs an invariant canonical SHA-256 hash key from normalized symptoms."""
        clean_present = sorted(set(self._normalize_token(s) for s in present_symptoms if s.strip()))
        clean_denied = sorted(set(self._normalize_token(s) for s in (denied_symptoms or []) if s.strip()))
        age_bucket = self._categorize_age(age)
        sex_bucket = (biological_sex or "unspecified").strip().lower()

        raw_repr = (
            f"canonical_present:{','.join(clean_present)}|"
            f"canonical_denied:{','.join(clean_denied)}|"
            f"age:{age_bucket}|"
            f"sex:{sex_bucket}"
        )
        return hashlib.sha256(raw_repr.encode("utf-8")).hexdigest()

    # Alias for backwards compatibility
    make_key = make_canonical_key

    def get(self, key: Optional[str]) -> Optional[DiagnosticReportSchema]:
        """Look up precomputed report by key. Returns None on cache miss."""
        if not key:
            return None

        with self._lock:
            if key in self._cache:
                self._hits += 1
                report, created_at = self._cache.pop(key)
                self._cache[key] = (report, created_at)  # Move to most recently used
                
                # Return a copy marked with cached=True
                report_copy = report.model_copy(deep=True)
                report_copy.cached = True
                return report_copy

            self._misses += 1
            return None

    def set(self, key: str, report: DiagnosticReportSchema) -> None:
        """Store diagnostic report in cache, evicting oldest item if exceeding max_size."""
        if not key:
            return

        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Evict least recently used

            self._cache[key] = (report, time.time())

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Return operational cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "total_queries": total,
                "hit_rate_percentage": round(hit_rate * 100.0, 2),
            }

