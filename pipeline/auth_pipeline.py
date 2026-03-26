"""
auth_pipeline.py
────────────────
End-to-end authentication: capture → detect → liveness → embed → verify → decide.
"""


def run_authentication(user_id: str, video_path: str | None = None) -> dict:
    """
    Authenticates a user against their stored identity.

    Returns a verdict dict: {"accepted": bool, "similarity": float,
                              "liveness_score": float, "reason": str}
    """
    raise NotImplementedError
