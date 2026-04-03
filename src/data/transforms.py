"""Augmentation pipelines for training and validation/test splits."""

from torchvision import transforms

# ImageNet statistics — reasonable starting point for pre-trained backbones.
# Replace with dataset-specific stats if you re-calculate them.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Heavy augmentation pipeline for the training split.

    Strategy:
    - Spatial: random resized crop + horizontal/vertical flip + rotation
    - Colour: jitter + grayscale (simulates varied lighting)
    - Regularisation: RandomErasing (cutout-style occlusion)
    - Normalisation: ImageNet statistics (suitable for transfer learning)
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
            # Random erasing after tensor conversion
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ]
    )


def get_val_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic pipeline for validation and test splits."""
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),  # ~256 for size=224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )
