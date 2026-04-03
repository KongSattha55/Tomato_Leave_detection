"""Trainer class — encapsulates the full training + validation loop."""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint

log = logging.getLogger(__name__)


class Trainer:
    """Manages training, validation, and checkpointing for a single model.

    Improvements over the original notebook approach:
    - AdamW with weight-decay (better generalisation than plain Adam)
    - Cosine annealing LR schedule (smooth decay, no manual tuning)
    - Mixed-precision (AMP) for ~2× speed on GPU with no accuracy cost
    - Label smoothing in CrossEntropyLoss (reduces overconfidence)
    - Gradient clipping (prevents exploding gradients)
    - EarlyStopping + ModelCheckpoint callbacks
    - Clean history dict returned for downstream plotting

    Args:
        model:          ``nn.Module`` to train.
        loaders:        Dict with keys ``"train"``, ``"val"`` (and optionally ``"test"``).
        num_classes:    Number of output classes.
        epochs:         Maximum training epochs.
        lr:             Initial learning rate.
        weight_decay:   L2 regularisation weight.
        label_smoothing: Smoothing factor for CrossEntropyLoss (0 = no smoothing).
        patience:       EarlyStopping patience in epochs.
        save_dir:       Directory for model checkpoints.
        model_name:     Checkpoint filename prefix.
        device:         ``"cuda"``, ``"mps"``, or ``"cpu"`` (auto-detected if ``None``).
    """

    def __init__(
        self,
        model: nn.Module,
        loaders: Dict[str, DataLoader],
        num_classes: int = 10,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        patience: int = 7,
        save_dir: str = "saved_models",
        model_name: str = "model",
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        log.info("Using device: %s", self.device)

        self.model = model.to(self.device)
        self.loaders = loaders
        self.epochs = epochs

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # AdamW — separates weight decay from adaptive momentum
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Cosine annealing over total epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 1e-2
        )

        # Mixed precision (only on CUDA)
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        self.early_stop = EarlyStopping(patience=patience, mode="min")
        self.checkpoint = ModelCheckpoint(
            save_dir=save_dir, model_name=model_name, mode="min"
        )

        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [],   "val_acc": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, list]:
        """Run the full training loop.

        Returns:
            History dict with per-epoch metrics.
        """
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(epoch, phase="train")
            val_loss, val_acc = self._run_epoch(epoch, phase="val")

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            log.info(
                "Epoch %02d/%02d  |  "
                "train loss=%.4f acc=%.3f  |  "
                "val loss=%.4f acc=%.3f  |  "
                "lr=%.2e  |  %.1fs",
                epoch, self.epochs,
                train_loss, train_acc,
                val_loss, val_acc,
                current_lr, elapsed,
            )

            self.checkpoint(val_loss, self.model, epoch)
            self.early_stop(val_loss)
            if self.early_stop.should_stop:
                log.info("Early stopping at epoch %d.", epoch)
                break

        # Reload best weights before returning
        if self.checkpoint.best_path and self.checkpoint.best_path.exists():
            self.model.load_state_dict(
                torch.load(self.checkpoint.best_path, map_location=self.device)
            )
            log.info("Restored best weights from %s", self.checkpoint.best_path)

        return self.history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int, phase: str) -> tuple[float, float]:
        is_train = phase == "train"
        self.model.train(is_train)
        loader = self.loaders[phase]

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"[{phase:5s}] E{epoch:02d}", leave=False)
        with torch.set_grad_enabled(is_train):
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with autocast(enabled=(self.device.type == "cuda")):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    # Gradient clipping — prevents exploding gradients
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size

                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / total
        avg_acc = correct / total
        return avg_loss, avg_acc
