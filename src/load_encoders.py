"""Load and freeze the three Phase 1-3 encoders, and verify each satisfies the
fusion contract:  forward(x) -> [B, 1024], no final activation, no head.

    python -m src.load_encoders          # runs the verification report

Public API used by fusion_train.py / fusion_demo.py:
    load_all_encoders(device, freeze=True) -> {name: EncoderBundle}
    embed(bundle, x) -> [B, 1024]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from . import fusion_config as FC
from .utils import setup_logging

logger = setup_logging()


@dataclass
class EncoderBundle:
    name: str
    encoder: nn.Module
    input_shape: tuple           # per-sample, no batch dim
    ckpt_path: Path
    ckpt_meta: dict              # selected fields from the checkpoint (labels, norm stats, ...)
    frozen: bool


# --------------------------------------------------------------------------- #
# Per-modality loaders  (each returns (encoder, meta_dict))
# --------------------------------------------------------------------------- #
def _load_xray(ckpt_path: Path, device):
    """Phase 1 has no `load_encoder_from_checkpoint` helper -> extract the
    `encoder.*` sub-state of MultiLabelClassifier into a bare XrayEncoder."""
    from .model import XrayEncoder

    ckpt = torch.load(ckpt_path, map_location=device)
    enc = XrayEncoder(pretrained=False)
    sub = {k[len("encoder."):]: v for k, v in ckpt["model_state"].items()
           if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(sub, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"xray encoder state mismatch: missing={missing} unexpected={unexpected}")
    meta = {
        "arch": ckpt.get("arch", "densenet121"),
        "orig_labels": ckpt.get("target_labels"),
        "orig_task": "multi-label (Cardiomegaly, Effusion)",
        "image_size": ckpt.get("image_size", 224),
        "val_metric": (ckpt.get("val_auroc") or {}).get("mean"),
    }
    return enc, meta


def _load_echo(ckpt_path: Path, device):
    from .echo_model import load_encoder_from_checkpoint

    enc, ckpt = load_encoder_from_checkpoint(ckpt_path, map_location=device)
    meta = {
        "arch": ckpt.get("arch", "resnet18_bilstm"),
        "orig_labels": ckpt.get("class_names"),
        "orig_task": "3-class EF category",
        "frames_per_clip": ckpt.get("frames_per_clip"),
        "norm_mean": ckpt.get("norm_mean"), "norm_std": ckpt.get("norm_std"),
        "val_metric": (ckpt.get("val_metrics") or {}).get("macro_auroc"),
    }
    return enc, meta


def _load_mri(ckpt_path: Path, device):
    from .mri_model import load_encoder_from_checkpoint

    enc, ckpt = load_encoder_from_checkpoint(ckpt_path, map_location=device)
    meta = {
        "arch": ckpt.get("arch", "resnet18_bilstm"),
        "orig_labels": ckpt.get("class_names"),
        "orig_task": "5-class diagnosis",
        "n_slices": ckpt.get("n_slices"),
        "norm_mean": ckpt.get("norm_mean"), "norm_std": ckpt.get("norm_std"),
        "val_metric": (ckpt.get("val_metrics") or {}).get("macro_auroc"),
    }
    return enc, meta


_LOADERS = {"xray": _load_xray, "echo": _load_echo, "mri": _load_mri}
_CKPTS = {"xray": FC.XRAY_CKPT, "echo": FC.ECHO_CKPT, "mri": FC.MRI_CKPT}


# --------------------------------------------------------------------------- #
# Freeze + public loader
# --------------------------------------------------------------------------- #
def freeze_encoder(enc: nn.Module) -> None:
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()


def load_all_encoders(device: Optional[torch.device] = None, freeze: bool = True) -> Dict[str, EncoderBundle]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundles: Dict[str, EncoderBundle] = {}
    for name in FC.MODALITIES:
        path = _CKPTS[name]
        if not path.exists():
            raise FileNotFoundError(f"{name} checkpoint missing: {path}")
        enc, meta = _LOADERS[name](path, device)
        enc.to(device)
        if freeze:
            freeze_encoder(enc)
        bundles[name] = EncoderBundle(
            name=name, encoder=enc, input_shape=FC.INPUT_SHAPES[name],
            ckpt_path=path, ckpt_meta=meta, frozen=freeze,
        )
    return bundles


@torch.no_grad()
def embed(bundle: EncoderBundle, x: torch.Tensor) -> torch.Tensor:
    """Raw modality input [B, *input_shape] -> [B, 1024] embedding."""
    return bundle.encoder(x)


# --------------------------------------------------------------------------- #
# Contract verification
# --------------------------------------------------------------------------- #
def verify_contract(bundle: EncoderBundle, device: torch.device, batch: int = 2) -> dict:
    x = torch.randn(batch, *bundle.input_shape, device=device)
    with torch.no_grad():
        y = bundle.encoder(x)

    n_param = sum(p.numel() for p in bundle.encoder.parameters())
    n_train = sum(p.numel() for p in bundle.encoder.parameters() if p.requires_grad)
    feat_dim_attr = getattr(bundle.encoder, "feature_dim", None)

    checks = {
        "output_shape": tuple(y.shape),
        "shape_ok": tuple(y.shape) == (batch, FC.EMBED_DIM),
        "feature_dim_attr": feat_dim_attr,
        "feature_dim_ok": feat_dim_attr == FC.EMBED_DIM,
        "finite": bool(torch.isfinite(y).all()),
        # a raw (un-activated) feature vector is not bounded to [0,1] or [-1,1];
        # report the range so it's visibly not a squashed activation
        "out_min": float(y.min()), "out_max": float(y.max()),
        "out_mean": float(y.mean()), "out_std": float(y.std()),
        "looks_unactivated": bool(y.min() < -0.05 or y.max() > 1.05),
        "params": n_param, "trainable_params": n_train,
        "frozen": n_train == 0,
    }
    return checks


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 74)
    logger.info("PHASE 4 — FROZEN ENCODER LOADING + [B,1024] CONTRACT VERIFICATION")
    logger.info("device: %s", device)
    logger.info("=" * 74)

    bundles = load_all_encoders(device, freeze=True)
    all_ok = True
    for name in FC.MODALITIES:
        b = bundles[name]
        c = verify_contract(b, device)
        ok = c["shape_ok"] and c["feature_dim_ok"] and c["finite"] and c["frozen"] and c["looks_unactivated"]
        all_ok &= ok
        logger.info("-" * 74)
        logger.info("[%s]  %s", name.upper(), b.ckpt_path.name)
        logger.info("  original task     : %s  (labels: %s)",
                    b.ckpt_meta.get("orig_task"), b.ckpt_meta.get("orig_labels"))
        logger.info("  original val AUROC : %s", _fmt(b.ckpt_meta.get("val_metric")))
        logger.info("  input  [B,%s]  ->  output %s",
                    ",".join(map(str, b.input_shape)), c["output_shape"])
        logger.info("  feature_dim attr  : %s  (expected %d)  %s",
                    c["feature_dim_attr"], FC.EMBED_DIM, "OK" if c["feature_dim_ok"] else "FAIL")
        logger.info("  params            : %s total, %s trainable  -> %s",
                    f"{c['params']:,}", f"{c['trainable_params']:,}",
                    "FROZEN" if c["frozen"] else "NOT FROZEN")
        logger.info("  output stats      : min=%.3f max=%.3f mean=%.3f std=%.3f",
                    c["out_min"], c["out_max"], c["out_mean"], c["out_std"])
        logger.info("  no final activation: %s (range is not squashed to [0,1]/[-1,1])",
                    "yes" if c["looks_unactivated"] else "SUSPECT")
        logger.info("  finite            : %s", c["finite"])
        logger.info("  CONTRACT: %s", "PASS" if ok else "*** FAIL ***")

    logger.info("=" * 74)
    logger.info("ALL THREE ENCODERS: %s", "PASS — ready for fusion" if all_ok else "*** ONE OR MORE FAILED ***")
    logger.info("NOTE: encoders come from 3 disjoint datasets — no shared patients. "
                "Fusion is validated single-modality-present only; multi-modal combos are SYNTHETIC.")
    if not all_ok:
        raise SystemExit(1)


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


if __name__ == "__main__":
    main()
