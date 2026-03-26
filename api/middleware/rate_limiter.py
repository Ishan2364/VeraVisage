"""
rate_limiter.py
───────────────
Limits repeated authentication attempts to prevent brute-force spoofing.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Rejects requests exceeding the configured threshold per IP."""

    async def dispatch(self, request: Request, call_next):
        raise NotImplementedError
