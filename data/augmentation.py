"""
augmentation.py
───────────────
Albumentations-based augmentation pipelines for training and validation.
"""

def get_train_transforms(image_size: int = 112):
    """Returns heavy augmentation transforms for training."""
    raise NotImplementedError

def get_val_transforms(image_size: int = 112):
    """Returns light, deterministic transforms for validation/inference."""
    raise NotImplementedError
