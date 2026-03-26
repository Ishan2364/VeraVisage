"""
scripts/train_cnn_classifier.py
────────────────────────────────────────────────────────────────────────────
Fine-tune MobileNetV2 on LCC-FASD for binary live/spoof classification.

APPROACH
─────────
1. Load MobileNetV2 pretrained on ImageNet (instant, no download needed
   if torchvision is installed — weights are cached automatically)
2. Freeze all layers except the final classifier
3. Replace final layer with a 2-class output (live=1, spoof=0)
4. Fine-tune for 10 epochs on LCC-FASD training split
5. Validate on development split after each epoch
6. Save best model weights to data/models/mobilenetv2_spoof.pth

EXPECTED RESULTS (RTX 4050)
────────────────────────────
  Training time:  ~5-8 minutes
  Dev accuracy:   ~90-98%
  Inference time: ~2ms per frame on GPU

RUN
───
    python scripts/train_cnn_classifier.py
    python scripts/train_cnn_classifier.py --epochs 15  # more epochs
    python scripts/train_cnn_classifier.py --unfreeze    # fine-tune all layers
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from infrastructure.logger import get_logger

log = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "lcc_fasd"
MODEL_OUTPUT = PROJECT_ROOT / "data" / "models" / "mobilenetv2_spoof.pth"

SPLITS = {
    "train": DATASET_ROOT / "LCC_FASD_training",
    "dev":   DATASET_ROOT / "LCC_FASD_development",
    "eval":  DATASET_ROOT / "LCC_FASD_evaluation",
}

# ── Dataset class ─────────────────────────────────────────────────────────

class FaceAntiSpoofDataset(Dataset):
    """
    Loads LCC-FASD images from real/ and spoof/ subfolders.

    Labels:
        real/  → 1 (live)
        spoof/ → 0 (spoof)
    """

    def __init__(
        self,
        split_dir: Path,
        transform=None,
        max_per_class: int | None = None,
    ):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for label, folder_name in [(1, "real"), (0, "spoof")]:
            folder = split_dir / folder_name
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder}")

            paths = sorted(
                list(folder.glob("*.png")) +
                list(folder.glob("*.jpg")) +
                list(folder.glob("*.jpeg"))
            )

            if max_per_class:
                # Even sampling across the folder
                step = max(1, len(paths) // max_per_class)
                paths = paths[::step][:max_per_class]

            for p in paths:
                self.samples.append((p, label))

        # Shuffle so batches have mixed live/spoof
        import random
        random.shuffle(self.samples)

        n_real  = sum(1 for _, l in self.samples if l == 1)
        n_spoof = sum(1 for _, l in self.samples if l == 0)
        log.info(
            "Dataset from %s — real=%d, spoof=%d, total=%d",
            split_dir.name, n_real, n_spoof, len(self.samples),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # Return a black image on read failure rather than crashing
            image = Image.new("RGB", (224, 224), 0)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ── Transforms ────────────────────────────────────────────────────────────

def get_train_transform():
    """Heavy augmentation for training — prevents overfitting."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.2, hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transform():
    """Minimal transform for validation — deterministic."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ── Model ─────────────────────────────────────────────────────────────────

def build_model(unfreeze_all: bool = False) -> nn.Module:
    """
    Build MobileNetV2 with a binary classification head.

    Args:
        unfreeze_all: If True, fine-tune ALL layers.
                      If False (default), freeze backbone, train head only.
                      Unfreezing all layers gives better accuracy but
                      takes longer and needs a lower learning rate.

    Returns:
        PyTorch model ready for training.
    """
    # Load ImageNet pretrained weights
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model   = models.mobilenet_v2(weights=weights)

    if not unfreeze_all:
        # Freeze all backbone parameters — only train the classifier head
        for param in model.features.parameters():
            param.requires_grad = False
        log.info("Backbone frozen — training classifier head only")
    else:
        log.info("All layers unfrozen — full fine-tuning")

    # Replace the default 1000-class ImageNet head with a 2-class head
    # MobileNetV2's classifier is: Dropout → Linear(1280, 1000)
    # We replace the Linear layer: Linear(1280, 2)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)

    log.info(
        "MobileNetV2 ready — in_features=%d, out_features=2 (live/spoof)",
        in_features,
    )
    return model


# ── Training loop ─────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Run one training epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", unit="batch")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc":  f"{correct/total:.3f}",
        })

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
) -> tuple[float, float]:
    """Evaluate model on a dataloader. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc=f"  [{split_name}]", unit="batch"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total

    # Per-class accuracy
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    live_mask  = all_labels == 1
    spoof_mask = all_labels == 0

    live_acc  = (all_preds[live_mask]  == 1).mean() if live_mask.any()  else 0.0
    spoof_acc = (all_preds[spoof_mask] == 0).mean() if spoof_mask.any() else 0.0

    log.info(
        "%s — loss=%.4f acc=%.4f live_acc=%.4f spoof_acc=%.4f",
        split_name, total_loss / total, acc, live_acc, spoof_acc,
    )
    return total_loss / total, acc


# ── Main ──────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    # ── Datasets ──────────────────────────────────────────────────────────
    print("\n── Loading datasets ─────────────────────────────────────")
    train_ds = FaceAntiSpoofDataset(
        SPLITS["train"],
        transform=get_train_transform(),
        max_per_class=args.max_per_class,
    )
    dev_ds = FaceAntiSpoofDataset(
        SPLITS["dev"],
        transform=get_val_transform(),
        max_per_class=args.max_per_class,
    )

    # Compute class weights to handle imbalance (more spoof than real)
    n_real  = sum(1 for _, l in train_ds.samples if l == 1)
    n_spoof = sum(1 for _, l in train_ds.samples if l == 0)
    total   = n_real + n_spoof
    # Higher weight for the minority class (real faces)
    class_weights = torch.tensor(
        [total / (2 * n_spoof), total / (2 * n_real)],
        dtype=torch.float32,
    ).to(device)
    log.info(
        "Class weights — spoof=%.3f, live=%.3f",
        class_weights[0].item(), class_weights[1].item(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,       # 0 on Windows to avoid multiprocessing issues
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n── Building model ───────────────────────────────────────")
    model = build_model(unfreeze_all=args.unfreeze).to(device)

    # Weighted cross-entropy handles class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Adam with a small LR for fine-tuning
    # If unfreezing all layers, use an even smaller LR for backbone
    if args.unfreeze:
        optimizer = optim.Adam([
            {"params": model.features.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ])
    else:
        optimizer = optim.Adam(
            model.classifier.parameters(), lr=1e-3
        )

    # Reduce LR if dev accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5, verbose=True,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    print("\n── Training ─────────────────────────────────────────────")
    best_dev_acc  = 0.0
    best_epoch    = 0
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        dev_loss, dev_acc = evaluate(
            model, dev_loader, criterion, device, "dev"
        )
        scheduler.step(dev_acc)

        print(
            f"  Epoch {epoch:02d}/{args.epochs} — "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f}"
        )

        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch   = epoch
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "dev_acc":    dev_acc,
                "train_acc":  train_acc,
            }, MODEL_OUTPUT)
            print(f"  ✓ Best model saved (dev_acc={dev_acc:.4f})")

    # ── Final evaluation on eval split ────────────────────────────────────
    print("\n── Final evaluation on eval split ───────────────────────")
    eval_ds = FaceAntiSpoofDataset(
        SPLITS["eval"],
        transform=get_val_transform(),
        max_per_class=args.max_per_class,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    # Load best weights for final eval
    checkpoint = torch.load(MODEL_OUTPUT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    eval_loss, eval_acc = evaluate(
        model, eval_loader, criterion, device, "eval"
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MobileNetV2 Training Complete")
    print("=" * 60)
    print(f"  Best dev accuracy  : {best_dev_acc:.4f} (epoch {best_epoch})")
    print(f"  Eval accuracy      : {eval_acc:.4f}")
    print(f"  Model saved to     : {MODEL_OUTPUT}")
    print("=" * 60)
    print()
    print("  Next step: run test_phase3.py to verify liveness")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV2 on LCC-FASD for face anti-spoofing"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32, reduce to 16 if GPU OOM)",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        metavar="N",
        help="Max images per class per split. Use 500 for a quick test.",
    )
    parser.add_argument(
        "--unfreeze", action="store_true",
        help="Unfreeze all layers for full fine-tuning (slower, more accurate)",
    )
    main(parser.parse_args())