"""Model definitions.

`XrayEncoder` is deliberately split from the classification head so the same
backbone can later be dropped into a multi-modality fusion module (concatenate
its 1024-d feature with ECHO / MRI encoder outputs, then a shared head).

`build_model()` returns the full Phase-1 classifier (encoder + Linear(1024, 2))
with NO output activation - BCEWithLogitsLoss applies the sigmoid.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


DENSENET121_FEATURE_DIM = 1024


class XrayEncoder(nn.Module):
    """DenseNet121 feature extractor (ImageNet-pretrained), global-pooled.

    forward(x) -> (B, 1024). No classifier, no activation. This is the piece
    the fusion architecture will import.
    """

    feature_dim: int = DENSENET121_FEATURE_DIM

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import DenseNet121_Weights, densenet121

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        self.features = backbone.features  # conv stack, ends at norm5
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.relu(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class MultiLabelClassifier(nn.Module):
    """Encoder + linear head for N independent binary labels (logits out)."""

    def __init__(self, encoder: nn.Module, num_classes: int, feature_dim: Optional[int] = None) -> None:
        super().__init__()
        self.encoder = encoder
        dim = feature_dim or getattr(encoder, "feature_dim", DENSENET121_FEATURE_DIM)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# --------------------------------------------------------------------------- #
# Phase / freezing control
# --------------------------------------------------------------------------- #
def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def freeze_backbone(model: MultiLabelClassifier, train_last_block: bool = True) -> None:
    """Phase 1: freeze the DenseNet, optionally keep the final dense block +
    its norm trainable, always keep the classifier trainable.
    """
    _set_requires_grad(model.encoder, False)
    if train_last_block:
        feats = model.encoder.features
        for name, child in feats.named_children():
            if name in ("denseblock4", "norm5"):
                _set_requires_grad(child, True)
    _set_requires_grad(model.classifier, True)


def unfreeze_all(model: MultiLabelClassifier) -> None:
    """Phase 2: everything trainable."""
    _set_requires_grad(model, True)


def trainable_parameters(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(num_classes: int = 2, pretrained: bool = True) -> MultiLabelClassifier:
    encoder = XrayEncoder(pretrained=pretrained)
    return MultiLabelClassifier(encoder, num_classes=num_classes)
