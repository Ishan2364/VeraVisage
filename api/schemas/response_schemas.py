"""
response_schemas.py
───────────────────
Pydantic models for structuring all outgoing API response payloads.
"""
from pydantic import BaseModel


class EnrolmentResponse(BaseModel):
    user_id: str
    status: str
    detail: str


class AuthenticationResponse(BaseModel):
    accepted: bool
    similarity: float
    liveness_score: float
    reason: str
