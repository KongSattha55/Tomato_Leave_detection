"""Reusable plotting utilities for training diagnostics and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_history(
    history: Dict[str, List[float]],
    model_name: str = "model",
    save_dir: str | Path | None = None,
) -> None:
    """Plot training/validation loss, accuracy, and LR curves.

    Args:
        history:    Dict returned by :meth:`Trainer.fit`.
        model_name: Used in the plot title and saved filename.
        save_dir:   If given, saves the figure to ``{save_dir}/{model_name}_history.png``.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    has_lr = "lr" in history and len(history["lr"]) > 0
    n_rows = 3 if has_lr else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4 * n_rows))
    fig.suptitle(f"{model_name} — Training History", fontsize=13, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Acc")
    ax.plot(epochs, history["val_acc"], label="Val Acc")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    # Learning rate
    if has_lr:
        ax = axes[2]
        ax.plot(epochs, history["lr"], color="orange", label="LR")
        ax.set_ylabel("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        out = Path(save_dir) / f"{model_name}_history.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    model_name: str = "model",
    save_dir: str | Path | None = None,
    normalize: bool = True,
) -> None:
    """Plot a colour-coded confusion matrix and print a classification report.

    Args:
        y_true:       Ground-truth labels.
        y_pred:       Predicted labels.
        class_names:  Ordered list of class names.
        model_name:   Used in the title and saved filename.
        save_dir:     If given, saves the figure.
        normalize:    Normalise by true class counts (shows recall per class).
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_dir is not None:
        out = Path(save_dir) / f"{model_name}_confusion.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"\n{'='*60}")
    print(f"Classification Report — {model_name}")
    print('='*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def plot_sample_predictions(
    images: "np.ndarray",
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    n: int = 16,
    save_dir: str | Path | None = None,
    model_name: str = "model",
) -> None:
    """Display a grid of sample images with true/predicted labels.

    Args:
        images:      Array of shape (N, H, W, 3) in [0, 1].
        y_true:      True labels.
        y_pred:      Predicted labels.
        class_names: Class name list.
        n:           Number of images to show (clipped to 16).
        save_dir:    Optional save directory.
        model_name:  Figure title prefix.
    """
    n = min(n, len(images), 16)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        ax.imshow(np.clip(images[i], 0, 1))
        true_name = class_names[y_true[i]].replace("Tomato___", "")
        pred_name = class_names[y_pred[i]].replace("Tomato___", "")
        color = "green" if y_true[i] == y_pred[i] else "red"
        ax.set_title(f"T: {true_name}\nP: {pred_name}", fontsize=7, color=color)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{model_name} — Predictions", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_dir is not None:
        out = Path(save_dir) / f"{model_name}_predictions.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)
