"""Phase 2 model: CNN-per-frame + BiLSTM temporal aggregator.

`EchoEncoder` is the piece that survives into the fusion model. It matches the
Phase 1 contract (see PHASE1_XRAY_SUMMARY.md sec. 5):

    forward(x: [B, T, C, H, W]) -> [B, 1024]      # NO classifier, NO activation
    .feature_dim = 1024

`EchoClassifier` is a THROW-AWAY training wrapper: it bolts a `Linear(1024, 3)`
head on for standalone EF-category training this phase. Fusion code imports
`EchoEncoder` only (or calls `load_encoder_from_checkpoint`).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from . import echo_config as EC


class EchoEncoder(nn.Module):
    """ResNet18 (ImageNet) applied per frame -> BiLSTM over time -> temporal mean pool.

    ResNet18 gives 512-d per frame; a 1-layer **bidirectional** LSTM with hidden
    512 yields 2*512 = 1024 per timestep, mean-pooled across T -> [B, 1024].
    Mean pooling (vs. last hidden state) is robust to where in the cardiac cycle
    the sampled clip happens to start.
    """

    feature_dim: int = EC.D_ECHO  # 1024

    def __init__(self, pretrained: bool = True, rnn_hidden: int = EC.RNN_HIDDEN) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.cnn_feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.cnn = backbone  # [B*T, 3, H, W] -> [B*T, 512]

        self.rnn = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=rnn_hidden,
            num_layers=EC.RNN_LAYERS,
            batch_first=True,
            bidirectional=EC.RNN_BIDIRECTIONAL,
            dropout=EC.RNN_DROPOUT if EC.RNN_LAYERS > 1 else 0.0,
        )
        out_dim = rnn_hidden * (2 if EC.RNN_BIDIRECTIONAL else 1)
        assert out_dim == self.feature_dim, (
            f"RNN output {out_dim} != D_ECHO {self.feature_dim}; adjust RNN_HIDDEN/bidirectional"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        x = x.flatten(0, 1)                       # [B*T, C, H, W]
        feats = self.cnn(x)                       # [B*T, 512]
        feats = feats.view(b, t, self.cnn_feature_dim)
        seq, _ = self.rnn(feats)                  # [B, T, 1024]
        return seq.mean(dim=1)                    # [B, 1024] — no activation


class EchoClassifier(nn.Module):
    """TEMPORARY training wrapper: EchoEncoder + Linear head (logits, no softmax)."""

    def __init__(self, encoder: EchoEncoder, num_classes: int = EC.NUM_CLASSES) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# --------------------------------------------------------------------------- #
# Freeze control (two-phase training)
# --------------------------------------------------------------------------- #
def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def freeze_cnn(model: EchoClassifier) -> None:
    """Phase 1: freeze the per-frame CNN; LSTM + head stay trainable."""
    _set_requires_grad(model.encoder.cnn, False)
    _set_requires_grad(model.encoder.rnn, True)
    _set_requires_grad(model.head, True)


def unfreeze_all(model: EchoClassifier) -> None:
    _set_requires_grad(model, True)


def set_frozen_bn_eval(model: EchoClassifier) -> None:
    """Call after model.train() during the frozen phase so the CNN's BatchNorm
    running stats are NOT updated while its weights are frozen."""
    for m in model.encoder.cnn.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def trainable_parameters(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def build_echo_model(num_classes: int = EC.NUM_CLASSES, pretrained: bool = True) -> EchoClassifier:
    return EchoClassifier(EchoEncoder(pretrained=pretrained), num_classes=num_classes)


def load_encoder_from_checkpoint(path, map_location="cpu") -> Tuple[EchoEncoder, dict]:
    """Rebuild JUST the reusable EchoEncoder from an EchoClassifier checkpoint.
    This is what Phase 3 / the fusion model calls."""
    ckpt = torch.load(path, map_location=map_location)
    enc = EchoEncoder(pretrained=False)
    enc_state = {
        k[len("encoder."):]: v for k, v in ckpt["model_state"].items() if k.startswith("encoder.")
    }
    missing, unexpected = enc.load_state_dict(enc_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"encoder state mismatch: missing={missing}, unexpected={unexpected}")
    enc.eval()
    return enc, ckpt
