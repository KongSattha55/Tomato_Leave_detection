"""TomatoDataset — ImageFolder-compatible dataset for tomato leaf disease classification."""

from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class TomatoDataset(Dataset):
    """Loads tomato leaf images from a directory tree organised as:

        root/
            <class_name>/
                img1.jpg
                img2.jpg
                ...

    Supports optional transforms (albumentations or torchvision).
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        classes: Optional[list[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        # Discover classes from folder names, sorted for reproducibility
        if classes is not None:
            self.classes = sorted(classes)
        else:
            self.classes = sorted(
                d.name for d in self.root.iterdir() if d.is_dir()
            )

        self.class_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        self.samples: list[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, self.class_to_idx[cls]))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"TomatoDataset(root={self.root}, "
            f"classes={len(self.classes)}, "
            f"samples={len(self.samples)})"
        )
