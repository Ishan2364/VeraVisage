"""
scripts/train_texture_svm_lcc.py
────────────────────────────────────────────────────────────────────────────
Train the TextureAnalyzer SVM on the LCC-FASD dataset.

WHY LCC-FASD INSTEAD OF NUAA
──────────────────────────────
NUAA (2010) was collected with old VGA cameras. LCC-FASD (2020) uses
modern HD cameras in real-world conditions — matching your webcam's image
statistics. See project report Section 3.2 for full justification.

DATASET STRUCTURE EXPECTED
───────────────────────────
data/raw/lcc_fasd/
    LCC_FASD_training/
        real/    *.png   ← live faces,  label = 1
        spoof/   *.png   ← attack faces, label = 0
    LCC_FASD_development/
        real/    *.png
        spoof/   *.png
    LCC_FASD_evaluation/
        real/    *.png
        spoof/   *.png

RUN
───
    # Quick test (~3 min, 1000 images per class, no CV)
    python scripts/train_texture_svm_lcc.py --max-images 1000 --no-cv

    # Full training (~25-40 min, all images, with CV)
    python scripts/train_texture_svm_lcc.py

OUTPUT
──────
    data/models/texture_svm.pkl   ← overwrites any previous model
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

from infrastructure.logger import get_logger
from liveness.spatial.texture_analyzer import TextureAnalyzer

log = get_logger(__name__)

# ── Dataset paths ─────────────────────────────────────────────────────────
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "lcc_fasd"
MODEL_OUTPUT = PROJECT_ROOT / "data" / "models" / "texture_svm.pkl"

SPLITS = {
    "train": DATASET_ROOT / "LCC_FASD_training",
    "dev":   DATASET_ROOT / "LCC_FASD_development",
    "eval":  DATASET_ROOT / "LCC_FASD_evaluation",
}


def load_images_from_folder(
    folder: Path,
    label_name: str,
    max_images: int | None = None,
) -> list[np.ndarray]:
    """
    Load all PNG/JPG images from a folder as BGR NumPy arrays.

    Args:
        folder:     Path to real/ or spoof/ folder.
        label_name: For logging only ("real" or "spoof").
        max_images: Cap on images loaded. None = load all.

    Returns:
        List of BGR NumPy arrays.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            f"Make sure LCC_FASD is at: {DATASET_ROOT}\n"
            f"Run: Move-Item 'C:\\Users\\rocky\\Downloads\\archive (4)\\LCC_FASD' "
            f"'data\\raw\\lcc_fasd'"
        )

    # Collect all image files
    image_paths = sorted(
        list(folder.glob("*.png")) +
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {folder}")

    if max_images and max_images < len(image_paths):
        # Sample evenly so we get diverse subjects, not just first N files
        step = max(1, len(image_paths) // max_images)
        image_paths = image_paths[::step][:max_images]

    images = []
    failed = 0

    for path in tqdm(image_paths, desc=f"  {label_name}", unit="img"):
        img = cv2.imread(str(path))
        if img is None:
            failed += 1
            continue
        images.append(img)

    log.info(
        "Loaded %d %s images from %s (%d failed to read)",
        len(images), label_name, folder.name, failed,
    )
    return images


def load_split(
    split_name: str,
    max_images_per_class: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Load real and spoof images from one dataset split.

    Returns:
        Tuple of (real_images, spoof_images)
    """
    split_dir = SPLITS[split_name]
    log.info("Loading '%s' split from %s", split_name, split_dir)

    real_images  = load_images_from_folder(
        split_dir / "real",  "real",  max_images_per_class
    )
    spoof_images = load_images_from_folder(
        split_dir / "spoof", "spoof", max_images_per_class
    )
    return real_images, spoof_images


def evaluate_split(
    analyzer: TextureAnalyzer,
    real_images: list[np.ndarray],
    spoof_images: list[np.ndarray],
    split_name: str,
) -> dict:
    """Evaluate the trained SVM on a split and print a classification report."""
    from sklearn.metrics import classification_report, confusion_matrix

    all_images = real_images + spoof_images
    y_true = [1] * len(real_images) + [0] * len(spoof_images)
    y_pred = []

    for img in tqdm(all_images, desc=f"  Evaluating {split_name}", unit="img"):
        features = analyzer.extract_features(img).reshape(1, -1)
        y_pred.append(int(analyzer._svm.predict(features)[0]))

    report = classification_report(
        y_true, y_pred,
        target_names=["spoof", "live"],
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n── {split_name.upper()} RESULTS ───────────────────────────")
    print(classification_report(y_true, y_pred, target_names=["spoof", "live"]))
    print(f"Confusion matrix:\n{cm}")

    return {
        "accuracy":  round(report["accuracy"], 4),
        "precision": round(report["live"]["precision"], 4),
        "recall":    round(report["live"]["recall"], 4),
        "f1":        round(report["live"]["f1-score"], 4),
    }


def main(args: argparse.Namespace) -> None:

    # ── Verify dataset exists ─────────────────────────────────────────────
    if not DATASET_ROOT.exists():
        print(f"\nERROR: Dataset not found at {DATASET_ROOT}")
        print("Move it with PowerShell:")
        print(f'  Move-Item "C:\\Users\\rocky\\Downloads\\archive (4)\\LCC_FASD" '
              f'"data\\raw\\lcc_fasd"')
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  VeraVisage — Texture SVM Training on LCC-FASD")
    print("=" * 60)
    if args.max_images:
        print(f"  Mode: QUICK TEST ({args.max_images} images/class)")
    else:
        print("  Mode: FULL TRAINING (all images)")
    print(f"  Cross-validation: {'disabled' if args.no_cv else 'enabled (5-fold)'}")
    print("=" * 60 + "\n")

    # ── Load training split ───────────────────────────────────────────────
    print("Step 1/4 — Loading training images...")
    real_train, spoof_train = load_split("train", args.max_images)
    print(f"  real={len(real_train)}, spoof={len(spoof_train)}\n")

    # ── Train ─────────────────────────────────────────────────────────────
    print("Step 2/4 — Training SVM...")
    analyzer = TextureAnalyzer()

    if args.no_cv:
        # Fast path — fit only, no cross-validation
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        print("  Extracting LBP features...")
        X_real  = analyzer.extract_features_batch(real_train)
        X_spoof = analyzer.extract_features_batch(spoof_train)
        X = np.vstack([X_real, X_spoof])
        y = np.array([1] * len(real_train) + [0] * len(spoof_train))

        # WITH this — adds SMOTE oversampling + tuned C:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE

        analyzer._svm = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote",  SMOTE(random_state=42, k_neighbors=5)),
            ("svm",    SVC(
                kernel="rbf", C=100.0, gamma="scale",
                probability=True, class_weight="balanced",
            )),
        ])
        print("  Fitting RBF SVM (this is the slow step)...")
        analyzer._svm.fit(X, y)
        train_acc = analyzer._svm.score(X, y)
        print(f"  Train accuracy: {train_acc:.4f}")
        train_results = {"train_accuracy": round(train_acc, 4), "cv": "skipped"}

    else:
        train_results = analyzer.train(real_train, spoof_train)

    print(f"\n  Train results: {train_results}\n")

    # ── Evaluate on development set ───────────────────────────────────────
    print("Step 3/4 — Evaluating on development set...")
    real_dev, spoof_dev = load_split("dev", args.max_images)
    dev_results = evaluate_split(analyzer, real_dev, spoof_dev, "development")

    # ── Evaluate on evaluation set ────────────────────────────────────────
    print("\nStep 4/4 — Evaluating on evaluation set...")
    real_eval, spoof_eval = load_split("eval", args.max_images)
    eval_results = evaluate_split(analyzer, real_eval, spoof_eval, "evaluation")

    # ── Save model ────────────────────────────────────────────────────────
    analyzer.save_model(MODEL_OUTPUT)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    if not args.no_cv:
        print(f"  Train CV accuracy  : "
              f"{train_results['cv_accuracy_mean']:.3f} "
              f"± {train_results['cv_accuracy_std']:.3f}")
    else:
        print(f"  Train accuracy     : {train_results['train_accuracy']:.3f}")
    print(f"  Dev  accuracy      : {dev_results['accuracy']:.3f}")
    print(f"  Dev  F1 (live)     : {dev_results['f1']:.3f}")
    print(f"  Eval accuracy      : {eval_results['accuracy']:.3f}")
    print(f"  Eval F1 (live)     : {eval_results['f1']:.3f}")
    print(f"  Model saved to     : {MODEL_OUTPUT}")
    print("=" * 60)
    print()
    print("  Test it live:")
    print("  python test_phase3.py --with-texture")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TextureAnalyzer SVM on LCC-FASD"
    )
    parser.add_argument(
        "--max-images", type=int, default=None, metavar="N",
        help="Max images per class per split. Use 1000 for a quick test.",
    )
    parser.add_argument(
        "--no-cv", action="store_true",
        help="Skip 5-fold cross-validation (3-5x faster).",
    )
    main(parser.parse_args())