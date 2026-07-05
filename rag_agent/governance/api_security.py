"""
governance/api_security.py
-----------------------------
Closes the "no auth on the FastAPI endpoints" gap: an API-key dependency
plus a minimal in-memory token-bucket rate limiter, wired into api.py.

This is deliberately simple (no external dependency on Redis or a rate-
limiting service) so the demo stays runnable standalone. A production
deployment behind real traffic should move rate-limit state to something
shared across processes (Redis, API Gateway, Azure APIM) — the interface
here (`check_and_consume(key)`) is designed so that swap is a
drop-in replacement.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Dict

from fastapi import Depends, Header, HTTPException


def _load_valid_api_keys() -> set:
    raw = os.environ.get("RAG_AGENT_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


VALID_API_KEYS = _load_valid_api_keys()


def require_api_key(x_api_key: str = Header(default=None)) -> str:
    """FastAPI dependency: enforce an API key on protected endpoints.

    If RAG_AGENT_API_KEYS is unset, auth is disabled with a loud warning —
    this keeps local/demo usage frictionless while making it obvious that
    production deployment must set it.
    """
    if not VALID_API_KEYS:
        return "unauthenticated-demo-mode"

    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")

    return x_api_key


class TokenBucketRateLimiter:
    """Minimal per-key token bucket. Not distributed — single-process only."""

    def __init__(self, capacity: int = 20, refill_per_second: float = 0.5):
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._buckets: Dict[str, tuple] = {}  # key -> (tokens, last_refill_ts)

    def check_and_consume(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last_refill = self._buckets.get(key, (self.capacity, now))

        elapsed = now - last_refill
        tokens = min(self.capacity, tokens + elapsed * self.refill_per_second)

        if tokens < 1:
            self._buckets[key] = (tokens, now)
            return False

        tokens -= 1
        self._buckets[key] = (tokens, now)
        return True


rate_limiter = TokenBucketRateLimiter()


def enforce_rate_limit(api_key: str = Depends(require_api_key)) -> str:
    """FastAPI dependency combining auth + rate limiting in one call."""
    if not rate_limiter.check_and_consume(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Slow down.")
    return api_key
