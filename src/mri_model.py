"""Phase 3 model: 2D-CNN per slice + BiLSTM over the slice axis.

`MriEncoder` matches the fusion contract (PHASE1_XRAY_SUMMARY.md sec. 5,
PHASE2_ECHO_SUMMARY.md sec. 5) — same as `EchoEncoder`, slices instead of frames:

    forward(x: [B, N_SLICES, C, H, W]) -> [B, 1024]     # NO head, NO activation
    .feature_dim = 1024

`MriClassifier` is the throw-away training wrapper (Dropout + Linear(1024, 5)).
Fusion / Phase 4 imports `MriEncoder` only, or calls
`load_encoder_from_checkpoint`.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from . import mri_config as MC


class MriEncoder(nn.Module):
    """ResNet18 (ImageNet) per short-axis slice -> BiLSTM over slices ->
    temporal(=depth) mean pool -> [B, 1024]. Structurally identical to
    EchoEncoder; the sequence axis is slice index, not video time."""

    feature_dim: int = MC.D_MRI  # 1024

    def __init__(self, pretrained: bool = True, rnn_hidden: int = MC.RNN_HIDDEN) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.cnn_feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.cnn = backbone

        self.rnn = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=rnn_hidden,
            num_layers=MC.RNN_LAYERS,
            batch_first=True,
            bidirectional=MC.RNN_BIDIRECTIONAL,
            dropout=MC.RNN_DROPOUT if MC.RNN_LAYERS > 1 else 0.0,
        )
        out_dim = rnn_hidden * (2 if MC.RNN_BIDIRECTIONAL else 1)
        assert out_dim == self.feature_dim, f"RNN out {out_dim} != D_MRI {self.feature_dim}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape[:2]
        x = x.flatten(0, 1)                        # [B*N, C, H, W]
        feats = self.cnn(x).view(b, n, self.cnn_feature_dim)
        seq, _ = self.rnn(feats)                   # [B, N, 1024]
        return seq.mean(dim=1)                     # [B, 1024] — no activation


class MriClassifier(nn.Module):
    """TEMPORARY training wrapper: MriEncoder + Dropout + Linear (logits)."""

    def __init__(self, encoder: MriEncoder, num_classes: int = MC.NUM_CLASSES,
                 dropout: float = MC.HEAD_DROPOUT) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.encoder(x)))


# --------------------------------------------------------------------------- #
# Freeze control
# --------------------------------------------------------------------------- #
def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def freeze_cnn(model: MriClassifier) -> None:
    """Freeze the per-slice CNN; BiLSTM + head train. (ACDC has 70 train
    patients — fine-tuning 11M ResNet params is not viable.)"""
    _set_requires_grad(model.encoder.cnn, False)
    _set_requires_grad(model.encoder.rnn, True)
    _set_requires_grad(model.head, True)


def unfreeze_all(model: MriClassifier) -> None:
    _set_requires_grad(model, True)


def set_frozen_bn_eval(model: MriClassifier) -> None:
    for m in model.encoder.cnn.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def trainable_parameters(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return trn, tot


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def build_mri_model(num_classes: int = MC.NUM_CLASSES, pretrained: bool = True) -> MriClassifier:
    return MriClassifier(MriEncoder(pretrained=pretrained), num_classes=num_classes)


def load_encoder_from_checkpoint(path, map_location="cpu") -> Tuple[MriEncoder, dict]:
    """Rebuild JUST the reusable MriEncoder from an MriClassifier checkpoint —
    what the Phase 4 fusion model calls."""
    ckpt = torch.load(path, map_location=map_location)
    enc = MriEncoder(pretrained=False)
    enc_state = {
        k[len("encoder."):]: v for k, v in ckpt["model_state"].items() if k.startswith("encoder.")
    }
    missing, unexpected = enc.load_state_dict(enc_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"encoder state mismatch: missing={missing}, unexpected={unexpected}")
    enc.eval()
    return enc, ckpt
