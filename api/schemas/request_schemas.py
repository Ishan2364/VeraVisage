"""
request_schemas.py
──────────────────
Pydantic models for validating all incoming API request payloads.
"""
from pydantic import BaseModel


class EnrolmentRequest(BaseModel):
    user_id: str


class AuthenticationRequest(BaseModel):
    user_id: str
