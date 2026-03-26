"""
scripts/train_texture_svm.py
────────────────────────────────────────────────────────────────────────────
Train the TextureAnalyzer SVM on the NUAA dataset and save the model.

Run from the project root:
    python scripts/train_texture_svm.py

The trained model is saved to:
    data/models/texture_svm.pkl

This script only needs to be run ONCE. After that, TextureAnalyzer loads
the saved model automatically at inference time.
────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

# ── Make sure project root is on sys.path ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

from infrastructure.logger import get_logger
from infrastructure.config_loader import load_config
from liveness.spatial.texture_analyzer import TextureAnalyzer

log = get_logger(__name__)


# ── Path configuration ────────────────────────────────────────────────────
# Kaggle prefix that appears in the .txt files
KAGGLE_PREFIX = "/kaggle/input/nuaaaa/raw/"

# Your local path where the dataset lives
# This maps to: deepfake_auth/data/raw/nuaa/raw/
LOCAL_PREFIX = str(PROJECT_ROOT / "data" / "raw" / "nuaa" / "raw") + "/"

# The txt files list "ClientRaw/" but your folder is named "ClientFace/"
# This dict handles that mismatch automatically
FOLDER_REMAP = {
    "ClientRaw/":   "ClientFace/",
    "ImposterRaw/": "ImposterRaw/",  # this one matches
}

# Dataset txt file paths
NUAA_DIR = PROJECT_ROOT / "data" / "raw" / "nuaa" / "raw"
CLIENT_TRAIN    = NUAA_DIR / "client_train_raw.txt"
IMPOSTER_TRAIN  = NUAA_DIR / "imposter_train_raw.txt"
CLIENT_TEST     = NUAA_DIR / "client_test_raw.txt"
IMPOSTER_TEST   = NUAA_DIR / "imposter_test_raw.txt"

# Output model path
MODEL_OUTPUT = PROJECT_ROOT / "data" / "models" / "texture_svm.pkl"


def remap_path(kaggle_path: str) -> Path:
    """
    Convert a Kaggle absolute path to your local Windows path.

    Example:
        /kaggle/input/nuaaaa/raw/ClientRaw/0001/0001_00_00_01_0.jpg
        → C:/Users/rocky/.../data/raw/nuaa/raw/ClientFace/0001/0001_00_00_01_0.jpg
    """
    # Remove Kaggle prefix
    relative = kaggle_path.replace(KAGGLE_PREFIX, "")

    # Remap folder names that differ between txt files and actual disk
    for old_folder, new_folder in FOLDER_REMAP.items():
        if relative.startswith(old_folder):
            relative = relative.replace(old_folder, new_folder, 1)
            break

    local_path = Path(LOCAL_PREFIX) / relative
    return local_path


def load_images_from_txt(
    txt_path: Path,
    max_images: int | None = None,
    label_name: str = "images",
) -> list[np.ndarray]:
    """
    Read image paths from a NUAA txt file and load them as BGR crops.

    Args:
        txt_path:   Path to the .txt file listing image paths.
        max_images: Cap on how many images to load. None = load all.
                    Use during development to test quickly.
        label_name: Just for logging (e.g. "real" or "spoof").

    Returns:
        List of BGR NumPy arrays, each a face crop.
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Dataset txt file not found: {txt_path}")

    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if max_images:
        lines = lines[:max_images]

    images = []
    missing = 0

    log.info("Loading %d %s images from %s...", len(lines), label_name, txt_path.name)

    for line in tqdm(lines, desc=f"Loading {label_name}", unit="img"):
        local_path = remap_path(line)

        if not local_path.exists():
            missing += 1
            if missing <= 3:
                log.warning("Image not found: %s", local_path)
            continue

        img = cv2.imread(str(local_path))
        if img is None:
            missing += 1
            continue

        images.append(img)

    log.info(
        "Loaded %d/%d %s images (%d missing/unreadable)",
        len(images), len(lines), label_name, missing,
    )
    return images


def evaluate_on_test(
    analyzer: TextureAnalyzer,
    client_test: Path,
    imposter_test: Path,
) -> dict:
    """Run the trained model on the test split and report metrics."""
    from sklearn.metrics import classification_report, confusion_matrix

    log.info("Evaluating on test set...")

    real_test  = load_images_from_txt(client_test,   label_name="real_test")
    spoof_test = load_images_from_txt(imposter_test, label_name="spoof_test")

    all_images = real_test + spoof_test
    y_true = [1] * len(real_test) + [0] * len(spoof_test)

    y_pred = []
    y_scores = []

    for img in tqdm(all_images, desc="Evaluating", unit="img"):
        features = analyzer.extract_features(img).reshape(1, -1)
        pred  = int(analyzer._svm.predict(features)[0])
        score = float(analyzer._svm.predict_proba(features)[0][1])
        y_pred.append(pred)
        y_scores.append(score)

    report = classification_report(y_true, y_pred,
                                   target_names=["spoof", "live"],
                                   output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    log.info("\n%s", classification_report(y_true, y_pred,
                                           target_names=["spoof", "live"]))
    log.info("Confusion matrix:\n%s", cm)

    return {
        "accuracy":  report["accuracy"],
        "precision": report["live"]["precision"],
        "recall":    report["live"]["recall"],
        "f1":        report["live"]["f1-score"],
        "confusion_matrix": cm.tolist(),
    }


def main():
    log.info("=" * 60)
    log.info("Training TextureAnalyzer SVM on NUAA dataset")
    log.info("=" * 60)

    # ── Load training images ──────────────────────────────────────────────
    real_train  = load_images_from_txt(CLIENT_TRAIN,   label_name="real")
    spoof_train = load_images_from_txt(IMPOSTER_TRAIN, label_name="spoof")

    if not real_train or not spoof_train:
        log.error(
            "Could not load training images. "
            "Check that the NUAA dataset is at: %s",
            NUAA_DIR,
        )
        sys.exit(1)

    # ── Train ─────────────────────────────────────────────────────────────
    analyzer = TextureAnalyzer()
    train_results = analyzer.train(real_train, spoof_train)

    log.info("Training results: %s", train_results)

    # ── Evaluate on test set ──────────────────────────────────────────────
    test_results = evaluate_on_test(analyzer, CLIENT_TEST, IMPOSTER_TEST)
    log.info("Test results: %s", test_results)

    # ── Save model ────────────────────────────────────────────────────────
    analyzer.save_model(MODEL_OUTPUT)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Texture SVM Training Complete")
    print("=" * 60)
    print(f"  Train CV accuracy : {train_results['cv_accuracy_mean']:.3f} "
          f"± {train_results['cv_accuracy_std']:.3f}")
    print(f"  Test accuracy     : {test_results['accuracy']:.3f}")
    print(f"  Test F1 (live)    : {test_results['f1']:.3f}")
    print(f"  Model saved to    : {MODEL_OUTPUT}")
    print("=" * 60)
    print("\n  You can now use TextureAnalyzer in the liveness pipeline.")
    print("  Load it with: analyzer.load_model('data/models/texture_svm.pkl')\n")


if __name__ == "__main__":
    main()