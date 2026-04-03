"""Transfer-learning wrappers for torchvision pre-trained backbones."""

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet50_Weights,
)


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name:
            param.requires_grad = False


def _make_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    """Shared classification head: BN → Dropout → FC → ReLU → Dropout → FC."""
    return nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout / 2),
        nn.Linear(512, num_classes),
    )


# ---------------------------------------------------------------------------
# Individual model builders
# ---------------------------------------------------------------------------

def mobilenetv2(num_classes: int = 10, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    _freeze_backbone(model)
    in_features = model.classifier[1].in_features
    model.classifier = _make_head(in_features, num_classes, dropout)
    return model


def efficientnetb0(num_classes: int = 10, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    _freeze_backbone(model)
    in_features = model.classifier[1].in_features
    model.classifier = _make_head(in_features, num_classes, dropout)
    return model


def densenet121(num_classes: int = 10, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    _freeze_backbone(model)
    in_features = model.classifier.in_features
    model.classifier = _make_head(in_features, num_classes, dropout)
    return model


def resnet50(num_classes: int = 10, dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    """ResNet-50 with improved head — new backbone vs. original notebook."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    _freeze_backbone(model)
    in_features = model.fc.in_features
    model.fc = _make_head(in_features, num_classes, dropout)
    return model
