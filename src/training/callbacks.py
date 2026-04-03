"""Training callbacks: EarlyStopping and ModelCheckpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience:  Epochs to wait before stopping after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode:      ``"min"`` for loss, ``"max"`` for accuracy.
    """

    def __init__(
        self, patience: int = 7, min_delta: float = 1e-4, mode: str = "min"
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best: float | None = None
        self.should_stop = False

    def __call__(self, value: float) -> None:
        if self.best is None:
            self.best = value
            return

        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            log.debug("EarlyStopping counter %d / %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.should_stop = True
                log.info("Early stopping triggered.")

    def reset(self) -> None:
        self.counter = 0
        self.best = None
        self.should_stop = False


class ModelCheckpoint:
    """Save the model whenever the monitored metric improves.

    Args:
        save_dir:    Directory where checkpoints are written.
        model_name:  Prefix used in the filename.
        mode:        ``"min"`` for loss, ``"max"`` for accuracy.
        save_best_only: If True, keep only the best checkpoint.
    """

    def __init__(
        self,
        save_dir: str | Path,
        model_name: str,
        mode: str = "min",
        save_best_only: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.mode = mode
        self.save_best_only = save_best_only
        self.best: float | None = None
        self.best_path: Path | None = None

    def __call__(self, value: float, model: nn.Module, epoch: int) -> bool:
        """Save checkpoint if value improved.

        Returns:
            True if a checkpoint was saved.
        """
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "min" and value < self.best:
            improved = True
        elif self.mode == "max" and value > self.best:
            improved = True

        if improved:
            self.best = value
            path = self.save_dir / f"{self.model_name}_best.pth"
            torch.save(model.state_dict(), path)
            self.best_path = path
            log.info("Checkpoint saved  →  %s  (epoch %d, value=%.4f)", path, epoch, value)
            return True

        if not self.save_best_only:
            path = self.save_dir / f"{self.model_name}_epoch{epoch:03d}.pth"
            torch.save(model.state_dict(), path)

        return False
