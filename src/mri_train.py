"""Phase 3 training: ACDC 5-class diagnosis classifier (ResNet18 + BiLSTM).

Single phase, CNN FROZEN throughout (70 training patients cannot fine-tune 11M
ResNet params). Only the BiLSTM + head train. Overfitting mitigation: frozen
backbone, weight_decay 1e-3, head dropout 0.3, light augmentation, and
best-checkpoint on val macro AUROC (= de-facto early stopping over 25 epochs).

CrossEntropyLoss (5-class single-label; ACDC is 20/20/20/20/20 so no class
weights). last.pt every epoch; --resume restores everything.

    python -m src.mri_train
    python -m src.mri_train --resume
    python -m src.mri_train --quick_test
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import mri_config as MC
from .mri_dataset import make_mri_dataloaders
from .mri_model import (
    build_mri_model,
    count_parameters,
    freeze_cnn,
    set_frozen_bn_eval,
    trainable_parameters,
    unfreeze_all,
)
from .utils import multilabel_auroc, set_seed, setup_logging

logger = setup_logging()


def describe_device():
    import torch

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        logger.info("CUDA: %s | %.2f GB VRAM | cc %d.%d",
                    torch.cuda.get_device_name(0), p.total_memory / 1024**3, p.major, p.minor)
        return torch.device("cuda")
    logger.warning("CUDA NOT available - CPU (slow)")
    return torch.device("cpu")


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: List[str]) -> Dict:
    """probs: [N, C] softmax. macro/per-class OvR AUROC + accuracy + balanced acc."""
    y_true = np.asarray(y_true).astype(int)
    onehot = np.eye(len(class_names))[y_true]
    auroc = multilabel_auroc(onehot, probs, class_names)  # per-class OvR + "mean" == macro
    pred = probs.argmax(1)
    acc = float((pred == y_true).mean())
    recalls = [float((pred[y_true == c] == c).mean()) for c in range(len(class_names)) if (y_true == c).any()]
    return {"macro_auroc": auroc["mean"], "per_class_auroc": auroc,
            "accuracy": acc, "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan")}


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp, frozen, desc):
    import torch
    from tqdm import tqdm

    model.train()
    if frozen:
        set_frozen_bn_eval(model)
    running, n = 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = criterion(model(x), y)
        if amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / max(n, 1)


def evaluate_split(model, loader, criterion, device, class_names, amp, desc="val"):
    import torch
    from tqdm import tqdm

    model.eval()
    running, n = 0.0, 0
    P, T = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            running += loss.item() * x.size(0)
            n += x.size(0)
            P.append(torch.softmax(logits.float(), 1).cpu().numpy())
            T.append(y.cpu().numpy())
    probs = np.concatenate(P)
    bad = ~np.isfinite(probs).all(1)
    if bad.any():  # stray non-finite (rare GPU flukes) -> uniform, don't crash metrics
        logger.warning("%d/%d val rows had non-finite probs; set to uniform", int(bad.sum()), len(probs))
        probs[bad] = 1.0 / probs.shape[1]
    m = compute_metrics(np.concatenate(T), probs, class_names)
    m["loss"] = running / max(n, 1)
    return m


def _save(path: Path, *, model, optimizer, scaler, cfg, epoch, best_metric, val_metrics) -> None:
    import torch

    from .mri_dataset import load_norm_stats

    mean, std = load_norm_stats()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch, "global_epoch": epoch,
            "best_metric": best_metric, "ckpt_metric": MC.CKPT_METRIC,
            "val_metrics": val_metrics,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "class_names": list(cfg.class_names),
            "n_slices": cfg.n_slices, "slice_hw": cfg.slice_hw,
            "d_mri": cfg.d_mri, "arch": "resnet18_bilstm",
            "norm_mean": [float(x) for x in mean], "norm_std": [float(x) for x in std],
            "epochs": cfg.epochs, "freeze_cnn": cfg.freeze_cnn,
        },
        tmp,
    )
    tmp.replace(path)


METRICS_FIELDS = ["epoch", "lr", "train_loss", "val_loss",
                  "val_macro_auroc", "val_accuracy", "val_balanced_accuracy"]


def _init_metrics_csv(path: Path, class_names: List[str], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        return
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(METRICS_FIELDS + [f"val_auroc_{c}" for c in class_names])


def _append_metrics(path: Path, row: Dict, class_names: List[str]) -> None:
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=METRICS_FIELDS + [f"val_auroc_{c}" for c in class_names]).writerow(row)


def main(cfg: "MC.MriRunConfig", resume: bool = False) -> None:
    import torch

    set_seed(cfg.seed)
    device = describe_device()

    loaders = make_mri_dataloaders(cfg)
    for req in ("train", "val"):
        if loaders.get(req) is None:
            raise RuntimeError(f"no data for split '{req}'")

    model = build_mri_model(num_classes=cfg.num_classes, pretrained=True).to(device)
    if cfg.freeze_cnn:
        freeze_cnn(model)
    else:
        unfreeze_all(model)
    trn, tot = count_parameters(model)
    logger.info("trainable %s / %s params | CNN %s", f"{trn:,}", f"{tot:,}",
                "FROZEN" if cfg.freeze_cnn else "trainable")

    criterion = torch.nn.CrossEntropyLoss()
    amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=cfg.lr, weight_decay=cfg.weight_decay)
    class_names = list(cfg.class_names)

    best_metric, start_epoch = -1.0, 0
    if resume:
        if not MC.CKPT_LAST.exists():
            raise SystemExit(f"--resume: no checkpoint at {MC.CKPT_LAST}")
        ck = torch.load(MC.CKPT_LAST, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        if ck.get("scaler_state") is not None:
            scaler.load_state_dict(ck["scaler_state"])
        best_metric = ck.get("best_metric", -1.0)
        start_epoch = int(ck.get("epoch", 0))
        logger.info("RESUME from %s: %d/%d epochs done, best macro AUROC=%.4f",
                    MC.CKPT_LAST.name, start_epoch, cfg.epochs, best_metric)

    _init_metrics_csv(MC.METRICS_CSV, class_names, append=resume)
    logger.info("Metrics -> %s | epochs=%d lr=%.1e wd=%.1e", MC.METRICS_CSV, cfg.epochs, cfg.lr, cfg.weight_decay)

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, device,
                             scaler, amp, cfg.freeze_cnn, f"e{epoch}/{cfg.epochs} [train]")
        val = evaluate_split(model, loaders["val"], criterion, device, class_names, amp,
                             f"e{epoch}/{cfg.epochs} [val]")
        pc = val["per_class_auroc"]
        logger.info("epoch %02d | train_loss=%.4f val_loss=%.4f | macroAUROC=%.4f acc=%.4f balAcc=%.4f "
                    "| per-class %s | %.0fs",
                    epoch, tr, val["loss"], val["macro_auroc"], val["accuracy"], val["balanced_accuracy"],
                    " ".join(f"{c}={pc[c]:.3f}" for c in class_names), time.time() - t0)
        row = {"epoch": epoch, "lr": cfg.lr, "train_loss": round(tr, 6),
               "val_loss": round(val["loss"], 6), "val_macro_auroc": round(val["macro_auroc"], 6),
               "val_accuracy": round(val["accuracy"], 6),
               "val_balanced_accuracy": round(val["balanced_accuracy"], 6),
               **{f"val_auroc_{c}": round(pc[c], 6) for c in class_names}}
        _append_metrics(MC.METRICS_CSV, row, class_names)

        if np.isfinite(val["macro_auroc"]) and val["macro_auroc"] > best_metric:
            best_metric = val["macro_auroc"]
            _save(MC.CKPT_BEST, model=model, optimizer=optimizer, scaler=scaler, cfg=cfg,
                  epoch=epoch, best_metric=best_metric, val_metrics=val)
            logger.info("  ^ new best macro AUROC=%.4f -> saved %s", best_metric, MC.CKPT_BEST.name)
        _save(MC.CKPT_LAST, model=model, optimizer=optimizer, scaler=scaler, cfg=cfg,
              epoch=epoch, best_metric=best_metric, val_metrics=val)

    logger.info("Done. Best val macro AUROC = %.4f | best: %s | last: %s",
                best_metric, MC.CKPT_BEST, MC.CKPT_LAST)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ACDC 5-class MRI diagnosis classifier")
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--unfreeze_cnn", action="store_true", help="(not recommended) train the CNN too")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = MC.MriRunConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_amp:
        cfg.amp = False
    if args.unfreeze_cnn:
        cfg.freeze_cnn = False
    if args.quick_test:
        cfg.apply_quick_test()
    main(cfg, resume=args.resume)
