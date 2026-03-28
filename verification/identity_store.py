"""
verification/identity_store.py
────────────────────────────────────────────────────────────────────────────
Persistent storage for enrolled face embeddings.

STORAGE FORMAT
──────────────
Each enrolled user gets one file:
  data/embeddings_store/{user_id}.npy

The file contains a single np.ndarray of shape (512,) — the L2-normalised
ArcFace embedding averaged across their 30 enrolment frames.

WHY .npy FILES (NOT A DATABASE)
────────────────────────────────
For an academic project with a small number of users, .npy files are:
  - Zero dependency (just NumPy)
  - Human inspectable (can be loaded and examined in a notebook)
  - Fast (sub-millisecond load time)
  - Portable (copy the folder to move all enrollments)

In production you'd use a vector database (FAISS, Pinecone, Weaviate)
for 1-to-N search across millions of users. For 1-to-1 verification
(which is what VeraVisage does — user claims an identity, we verify),
.npy files are perfectly sufficient.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from infrastructure.exceptions import IdentityNotFoundError, EnrolmentError
from infrastructure.logger import get_logger

log = get_logger(__name__)


class IdentityStore:
    """
    Stores and retrieves L2-normalised face embeddings keyed by user_id.

    Each user_id maps to exactly one embedding file on disk.
    Re-enrolling a user overwrites their previous embedding.
    """

    def __init__(self, store_dir: str | Path | None = None):
        """
        Args:
            store_dir: Directory where .npy embedding files are saved.
                       Defaults to data/embeddings_store/ relative to
                       the project root.
        """
        if store_dir is None:
            # Auto-detect project root from this file's location
            project_root = Path(__file__).resolve().parents[1]
            store_dir    = project_root / "data" / "embeddings_store"

        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        log.info("IdentityStore initialised — store_dir=%s", self.store_dir)

    def _path_for(self, user_id: str) -> Path:
        """Return the .npy file path for a given user_id."""
        # Sanitise user_id to prevent directory traversal
        safe_id = "".join(
            c for c in user_id if c.isalnum() or c in ("_", "-")
        )
        if not safe_id:
            raise ValueError(f"Invalid user_id: '{user_id}'")
        return self.store_dir / f"{safe_id}.npy"

    def enrol(self, user_id: str, embedding: np.ndarray) -> None:
        """
        Save an embedding for a user.

        If the user already exists, their embedding is overwritten.
        This is intentional — re-enrolment should always use the
        most recent capture.

        Args:
            user_id:   Unique string identifier for the user.
                       Alphanumeric + underscore + hyphen only.
            embedding: L2-normalised np.ndarray shape (512,), float32.
                       Must come from EmbeddingGenerator.generate_batch().

        Raises:
            EnrolmentError: If the embedding cannot be saved.
            ValueError:     If embedding shape is wrong.
        """
        if embedding.shape != (512,):
            raise ValueError(
                f"Embedding must have shape (512,), got {embedding.shape}. "
                f"Use EmbeddingGenerator.generate_batch() to produce it."
            )

        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains NaN or Inf values.")

        path = self._path_for(user_id)

        try:
            np.save(str(path), embedding)
            log.info(
                "Enrolled user '%s' — embedding saved to %s",
                user_id, path.name,
            )
        except Exception as e:
            raise EnrolmentError(
                f"Failed to save embedding for '{user_id}': {e}"
            ) from e

    def retrieve(self, user_id: str) -> np.ndarray:
        """
        Load the stored embedding for a user.

        Args:
            user_id: The user's identifier (must be already enrolled).

        Returns:
            np.ndarray shape (512,), float32, L2-normalised.

        Raises:
            IdentityNotFoundError: If user_id has not been enrolled.
        """
        path = self._path_for(user_id)

        if not path.exists():
            raise IdentityNotFoundError(user_id=user_id)

        embedding = np.load(str(path)).astype(np.float32)

        log.debug(
            "Retrieved embedding for '%s' — shape=%s, norm=%.4f",
            user_id, embedding.shape, np.linalg.norm(embedding),
        )
        return embedding

    def delete(self, user_id: str) -> None:
        """
        Remove a user's embedding from the store.

        Args:
            user_id: The user to remove.

        Raises:
            IdentityNotFoundError: If user_id is not enrolled.
        """
        path = self._path_for(user_id)

        if not path.exists():
            raise IdentityNotFoundError(user_id=user_id)

        path.unlink()
        log.info("Deleted enrollment for user '%s'", user_id)

    def list_users(self) -> list[str]:
        """
        Return all enrolled user IDs.

        Returns:
            Sorted list of user_id strings (without .npy extension).
        """
        users = sorted(p.stem for p in self.store_dir.glob("*.npy"))
        log.debug("Enrolled users: %s", users)
        return users

    def is_enrolled(self, user_id: str) -> bool:
        """Return True if the user has an embedding on disk."""
        return self._path_for(user_id).exists()