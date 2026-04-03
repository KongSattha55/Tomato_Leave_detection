"""Build train / validation / test DataLoaders from directory trees."""

from pathlib import Path
from typing import Dict, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .dataset import TomatoDataset
from .transforms import get_train_transforms, get_val_transforms


def build_dataloaders(
    train_dir: str | Path,
    test_dir: str | Path,
    val_split: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    image_size: int = 224,
) -> Tuple[Dict[str, DataLoader], list[str]]:
    """Create train / val / test DataLoaders.

    Args:
        train_dir:   Directory containing per-class subdirectories for training.
        test_dir:    Directory containing per-class subdirectories for testing.
        val_split:   Fraction of training data used for validation.
        batch_size:  Mini-batch size for all loaders.
        num_workers: Parallel workers for data loading.
        seed:        Random seed for reproducible splits.
        image_size:  Square crop size passed to transform pipelines.

    Returns:
        loaders:  Dict with keys ``"train"``, ``"val"``, ``"test"``.
        classes:  Ordered list of class names.
    """
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    # Discover class names from the training directory
    classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())

    # Full training set (no transform yet — applied via Subset wrappers)
    full_train = TomatoDataset(train_dir, transform=None, classes=classes)

    # Stratified split into train / val indices
    labels = [label for _, label in full_train.samples]
    train_idx, val_idx = train_test_split(
        range(len(full_train)),
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )

    # Wrap Subsets with their respective transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_set = _TransformSubset(full_train, train_idx, train_transform)
    val_set = _TransformSubset(full_train, val_idx, val_transform)
    test_set = TomatoDataset(test_dir, transform=val_transform, classes=classes)

    pin = torch.cuda.is_available()

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=True,
        ),
        "val": DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        ),
        "test": DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        ),
    }

    return loaders, classes


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

class _TransformSubset(torch.utils.data.Dataset):
    """Apply a transform to a Subset of TomatoDataset."""

    def __init__(
        self,
        dataset: TomatoDataset,
        indices: list[int],
        transform,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
