#!/usr/bin/env python
"""Train a tomato leaf disease classifier.

Usage
-----
    # Train with default config (efficientnetb0):
    python scripts/train.py

    # Override model and epochs:
    python scripts/train.py --model resnet50 --epochs 20

    # Use a custom config file:
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import pickle
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data import build_dataloaders
from src.models import build_model
from src.training import Trainer
from src.utils import plot_history, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train tomato leaf disease classifier")
    p.add_argument("--config",  default="configs/config.yaml", help="Path to YAML config")
    p.add_argument("--model",   default=None, help="Override model name")
    p.add_argument("--epochs",  type=int, default=None, help="Override number of epochs")
    p.add_argument("--lr",      type=float, default=None, help="Override learning rate")
    p.add_argument("--batch",   type=int, default=None, help="Override batch size")
    p.add_argument("--device",  default=None, help="Force device (cuda/mps/cpu)")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.model:  cfg["model"]["name"] = args.model
    if args.epochs: cfg["training"]["epochs"] = args.epochs
    if args.lr:     cfg["training"]["lr"] = args.lr
    if args.batch:  cfg["data"]["batch_size"] = args.batch

    setup_logging(log_file=cfg["paths"].get("log_file"))

    # ---- Data --------------------------------------------------------
    loaders, classes = build_dataloaders(
        train_dir=cfg["data"]["train_dir"],
        test_dir=cfg["data"]["test_dir"],
        val_split=cfg["data"]["val_split"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["data"]["seed"],
        image_size=cfg["data"]["image_size"],
    )
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train batches: {len(loaders['train'])}  |  "
          f"Val batches: {len(loaders['val'])}  |  "
          f"Test batches: {len(loaders['test'])}")

    # ---- Model -------------------------------------------------------
    model_name = cfg["model"]["name"]
    model = build_model(
        name=model_name,
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.3),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}  |  Trainable params: {n_params:,}")

    # ---- Training ----------------------------------------------------
    trainer = Trainer(
        model=model,
        loaders=loaders,
        num_classes=cfg["model"]["num_classes"],
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        label_smoothing=cfg["training"].get("label_smoothing", 0.1),
        patience=cfg["training"]["patience"],
        save_dir=cfg["paths"]["save_dir"],
        model_name=model_name,
        device=args.device,
    )

    history = trainer.fit()

    # ---- Save history & plot -----------------------------------------
    save_dir = Path(cfg["paths"]["save_dir"])
    hist_path = save_dir / f"{model_name}_history.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)
    print(f"History saved → {hist_path}")

    plot_history(
        history,
        model_name=model_name,
        save_dir=cfg["paths"]["figures_dir"],
    )

    print("\nDone. Run  python scripts/evaluate.py --model", model_name, "  to evaluate.")


if __name__ == "__main__":
    main()
