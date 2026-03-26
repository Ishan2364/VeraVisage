"""
enroll.py
─────────
POST /enroll — receives an image/video and triggers the enrolment pipeline.
"""
from fastapi import APIRouter, UploadFile, File, Form

router = APIRouter(tags=["Enrolment"])


@router.post("/enroll")
async def enroll_user(user_id: str = Form(...), file: UploadFile = File(...)):
    """Enrols a new identity; returns status and enrolled user_id."""
    raise NotImplementedError
