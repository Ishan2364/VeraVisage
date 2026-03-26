"""
infrastructure/exceptions.py
────────────────────────────────────────────────────────────────────────────
Custom exception hierarchy for deepfake_auth.

WHY NAMED EXCEPTIONS MATTER
────────────────────────────
Compare these two approaches:

  BAD:   raise ValueError("No face detected in frame")
  GOOD:  raise FaceNotFoundError("No face detected in frame")

With the BAD approach, a caller has to catch ValueError — but ValueError
is raised by hundreds of things (bad int conversion, wrong list index...).
You can't tell what went wrong without reading the message string.

With the GOOD approach, a caller does:

    try:
        result = run_authentication(user_id, video_path)
    except LivenessFailedError:
        return {"accepted": False, "reason": "Spoof detected"}
    except FaceNotFoundError:
        return {"accepted": False, "reason": "No face in frame"}
    except IdentityNotFoundError:
        return {"accepted": False, "reason": "User not enrolled"}

Clean, readable, and each failure path is handled correctly.

HIERARCHY
─────────
DeepfakeAuthError          ← catch-all for any project error
├── FaceNotFoundError       ← detection found nothing
├── MultipleFacesError      ← detection found >1 face, ambiguous
├── LivenessFailedError     ← liveness check says it's a spoof
├── IdentityNotFoundError   ← user_id not in the store
└── EnrolmentError          ← couldn't save identity to store
────────────────────────────────────────────────────────────────────────────
"""


class DeepfakeAuthError(Exception):
    """
    Base class for all deepfake_auth project exceptions.

    Catching this catches every project-specific error at once.
    Use specific subclasses in production code; use this only in
    top-level handlers (e.g., the FastAPI exception handler) where
    you want to catch anything the pipeline raised.
    """


class FaceNotFoundError(DeepfakeAuthError):
    """
    Raised when the face detector finds no face in the input.

    Typical causes:
      - The frame is too dark or blurry
      - The person is too far from the camera
      - The frame was captured before the person was in position

    Where raised: core_vision/face_detector.py
    Where caught: pipeline/auth_pipeline.py, api/routers/authenticate.py
    """


class MultipleFacesError(DeepfakeAuthError):
    """
    Raised when the face detector finds more than one face in a frame
    and the system cannot determine which one to authenticate.

    Where raised: core_vision/face_detector.py
    Where caught: pipeline/auth_pipeline.py
    """


class LivenessFailedError(DeepfakeAuthError):
    """
    Raised when the liveness aggregator's confidence score falls below
    the configured threshold — i.e., the system believes the input is
    a photo, screen replay, or deepfake video rather than a live person.

    Carries the liveness score so callers can log it.

    Example:
        raise LivenessFailedError(score=0.23, threshold=0.50)

    Where raised: liveness/liveness_aggregator.py
    Where caught: pipeline/auth_pipeline.py
    """

    def __init__(
        self,
        score: float | None = None,
        threshold: float | None = None,
        message: str | None = None,
    ):
        self.score = score
        self.threshold = threshold

        if message:
            detail = message
        elif score is not None and threshold is not None:
            detail = (
                f"Liveness check failed: score {score:.3f} < "
                f"threshold {threshold:.3f} — spoof suspected."
            )
        else:
            detail = "Liveness check failed — spoof suspected."

        super().__init__(detail)


class IdentityNotFoundError(DeepfakeAuthError):
    """
    Raised when the identity store has no embedding for the requested
    user_id. This means the user has not been enrolled yet.

    Carries the user_id so the error message is self-documenting.

    Example:
        raise IdentityNotFoundError(user_id="alice_01")

    Where raised: verification/identity_store.py
    Where caught: pipeline/auth_pipeline.py, api/routers/authenticate.py
    """

    def __init__(self, user_id: str | None = None, message: str | None = None):
        self.user_id = user_id

        if message:
            detail = message
        elif user_id:
            detail = (
                f"No enrolled identity found for user_id='{user_id}'. "
                f"Run the enrolment pipeline first."
            )
        else:
            detail = "No enrolled identity found. Run the enrolment pipeline first."

        super().__init__(detail)


class EnrolmentError(DeepfakeAuthError):
    """
    Raised when the enrolment pipeline fails to persist an identity —
    e.g., a disk write error or an invalid embedding shape.

    Where raised: verification/identity_store.py, pipeline/enroll_pipeline.py
    Where caught: api/routers/enroll.py
    """