"""Improved Custom CNN with BatchNorm, residual skip connections, and SE attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * scale


class _ConvBnRelu(nn.Sequential):
    """Conv2d → BN → ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ResBlock(nn.Module):
    """Two-conv residual block with SE attention and optional projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _ConvBnRelu(in_ch, out_ch, stride=stride),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = _SEBlock(out_ch)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.se(self.body(x)) + self.shortcut(x), inplace=True)


class CustomCNN(nn.Module):
    """Compact residual CNN designed for 224×224 tomato leaf images.

    Architecture:
        Stem  : 3 → 32,  7×7, stride 2          → 112×112
        Stage 1: ResBlock 32 → 64,  stride 2     →  56×56
        Stage 2: ResBlock 64 → 128, stride 2     →  28×28
        Stage 3: ResBlock 128 → 256, stride 2    →  14×14
        Stage 4: ResBlock 256 → 512, stride 2    →   7×7
        Head  : GlobalAvgPool → Dropout → FC(num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # → 56×56
        )
        self.stages = nn.Sequential(
            _ResBlock(32, 64, stride=2),   # 28×28
            _ResBlock(64, 128, stride=2),  # 14×14
            _ResBlock(128, 256, stride=2), #  7×7
            _ResBlock(256, 512, stride=2), #  4×4  (for 224 input)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return self.classifier(x)
