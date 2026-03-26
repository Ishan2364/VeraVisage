"""
authenticate.py
───────────────
POST /authenticate — triggers the auth pipeline and returns the verdict.
"""
from fastapi import APIRouter, UploadFile, File, Form

router = APIRouter(tags=["Authentication"])


@router.post("/authenticate")
async def authenticate_user(user_id: str = Form(...), file: UploadFile = File(...)):
    """Verifies a live face against the stored identity; returns ACCEPT/REJECT."""
    raise NotImplementedError
