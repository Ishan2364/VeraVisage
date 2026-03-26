"""
identity_store.py
─────────────────
CRUD interface for enrolling and retrieving stored identity embeddings.
"""
import numpy as np
from pathlib import Path


class IdentityStore:
    """Persists and retrieves identity embeddings keyed by user ID."""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def enroll(self, user_id: str, embedding: np.ndarray) -> None:
        """Saves the embedding for a new or existing user."""
        raise NotImplementedError

    def retrieve(self, user_id: str) -> np.ndarray:
        """Loads and returns the stored embedding for the given user."""
        raise NotImplementedError

    def delete(self, user_id: str) -> None:
        """Removes a user's embedding from the store."""
        raise NotImplementedError

    def list_users(self) -> list[str]:
        """Returns all enrolled user IDs."""
        raise NotImplementedError
