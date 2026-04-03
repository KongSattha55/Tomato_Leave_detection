"""Central registry for creating models by name."""

import torch.nn as nn

from .custom_cnn import CustomCNN
from .transfer_models import densenet121, efficientnetb0, mobilenetv2, resnet50

_REGISTRY: dict[str, callable] = {
    "custom_cnn":    lambda cfg: CustomCNN(num_classes=cfg["num_classes"], dropout=cfg.get("dropout", 0.4)),
    "mobilenetv2":   lambda cfg: mobilenetv2(num_classes=cfg["num_classes"], dropout=cfg.get("dropout", 0.3), pretrained=cfg.get("pretrained", True)),
    "efficientnetb0":lambda cfg: efficientnetb0(num_classes=cfg["num_classes"], dropout=cfg.get("dropout", 0.3), pretrained=cfg.get("pretrained", True)),
    "densenet121":   lambda cfg: densenet121(num_classes=cfg["num_classes"], dropout=cfg.get("dropout", 0.3), pretrained=cfg.get("pretrained", True)),
    "resnet50":      lambda cfg: resnet50(num_classes=cfg["num_classes"], dropout=cfg.get("dropout", 0.3), pretrained=cfg.get("pretrained", True)),
}


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    """Instantiate a model by name.

    Args:
        name:        One of the registered model names (see :func:`list_models`).
        num_classes: Number of output classes.
        **kwargs:    Extra arguments forwarded to the builder (e.g. ``dropout``, ``pretrained``).

    Returns:
        An ``nn.Module`` ready for training.
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    cfg = {"num_classes": num_classes, **kwargs}
    return _REGISTRY[name](cfg)


def list_models() -> list[str]:
    """Return all registered model names."""
    return sorted(_REGISTRY.keys())
