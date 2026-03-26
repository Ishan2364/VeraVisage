"""
enroll_pipeline.py
──────────────────
End-to-end enrolment: capture → detect → align → embed → store identity.
"""


def run_enroll(user_id: str, image_path: str) -> dict:
    """
    Enrolls a new user by processing their image and storing the embedding.

    Returns a status dict: {"user_id": str, "status": "enrolled" | "failed", "detail": str}
    """
    raise NotImplementedError
