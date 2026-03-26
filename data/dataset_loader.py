"""
dataset_loader.py
─────────────────
PyTorch Dataset classes for loading aligned face images from disk.
"""
from pathlib import Path
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    """Loads face images and their identity labels from a directory tree."""

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Walks root and maps each image to an integer class label."""
        raise NotImplementedError("Implement _build_index")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError("Implement __getitem__")
