#!/usr/bin/env python
"""Evaluate a saved model on the test set.

Usage
-----
    python scripts/evaluate.py --model efficientnetb0
    python scripts/evaluate.py --model resnet50 --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from tqdm import tqdm

from src.data import build_dataloaders
from src.models import build_model
from src.utils import plot_confusion_matrix, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate tomato leaf disease classifier")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model",  required=True, help="Model name (must match a saved checkpoint)")
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Data
    loaders, classes = build_dataloaders(
        train_dir=cfg["data"]["train_dir"],
        test_dir=cfg["data"]["test_dir"],
        val_split=cfg["data"]["val_split"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["data"]["seed"],
        image_size=cfg["data"]["image_size"],
    )

    # Model + weights
    model = build_model(
        name=args.model,
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,  # weights loaded from checkpoint
        dropout=cfg["model"].get("dropout", 0.3),
    )
    ckpt_path = Path(cfg["paths"]["save_dir"]) / f"{args.model}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded weights from {ckpt_path}")

    # Inference
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in tqdm(loaders["test"], desc="Test inference"):
            images = images.to(device, non_blocking=True)
            preds = model(images).argmax(dim=1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(labels.tolist())

    # Results
    correct = sum(t == p for t, p in zip(all_true, all_pred))
    print(f"\nTest accuracy: {correct / len(all_true):.4f}  ({correct}/{len(all_true)})")

    plot_confusion_matrix(
        y_true=all_true,
        y_pred=all_pred,
        class_names=classes,
        model_name=args.model,
        save_dir=cfg["paths"]["figures_dir"],
        normalize=True,
    )


if __name__ == "__main__":
    main()
