"""Two-phase training for the Cardiomegaly / Effusion classifier.

Phase 1  : backbone frozen except final dense block + head, lr = 1e-4, 5 epochs
Phase 2  : whole network unfrozen,                          lr = 1e-5, 5 epochs

- best checkpoint  (densenet121_best.pt) : highest mean val AUROC across both
  labels, saved on any epoch of either phase it improves.
- last checkpoint  (densenet121_last.pt) : written every epoch; `--resume`
  restores model + optimizer + AMP scaler + epoch counter from it.
- per-epoch metrics appended to outputs/logs/metrics.csv.

    python -m src.train                # full run
    python -m src.train --resume       # continue an interrupted run
    python -m src.train --quick_test   # 500 train / 100 val, 1 epoch, smoke test
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config as C
from .dataset import make_dataloaders
from .model import (
    build_model,
    count_parameters,
    freeze_backbone,
    trainable_parameters,
    unfreeze_all,
)
from .utils import multilabel_auroc, set_seed, setup_logging

logger = setup_logging()


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def describe_device():
    import torch

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / (1024 ** 3)
        logger.info(
            "CUDA available: %s | %.1f GB VRAM | capability %d.%d",
            name, vram_gb, props.major, props.minor,
        )
        return torch.device("cuda"), name, vram_gb
    logger.warning("CUDA NOT available - running on CPU (this will be slow).")
    return torch.device("cpu"), "cpu", 0.0


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp, desc):
    import torch
    from tqdm import tqdm

    model.train()
    running, n = 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        running += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{running / max(n, 1):.4f}")
    return running / max(n, 1)


def evaluate_split(model, loader, criterion, device, label_names, amp, desc="val"):
    import torch
    from tqdm import tqdm

    model.eval()
    running, n = 0.0, 0
    all_scores: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            running += loss.item() * images.size(0)
            n += images.size(0)
            all_scores.append(torch.sigmoid(logits).float().cpu().numpy())
            all_targets.append(labels.float().cpu().numpy())

    scores = np.concatenate(all_scores)
    targets = np.concatenate(all_targets)
    auroc = multilabel_auroc(targets, scores, label_names)
    return {"loss": running / max(n, 1), "auroc": auroc, "scores": scores, "targets": targets}


# --------------------------------------------------------------------------- #
# Checkpoint / metrics IO
# --------------------------------------------------------------------------- #
def _save(path: Path, *, model, optimizer, scaler, cfg, global_epoch, phase,
          phase_offset, best_auroc, val_auroc) -> None:
    """Write a checkpoint with everything needed to (a) evaluate and (b) --resume."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": global_epoch,          # completed global epochs
            "global_epoch": global_epoch,
            "phase": phase,                 # phase that produced this checkpoint
            "phase_offset": phase_offset,   # epochs completed within that phase
            "best_auroc": best_auroc,
            "val_auroc": val_auroc,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "target_labels": list(cfg.target_labels),
            "image_size": cfg.image_size,
            "arch": "densenet121",
            "phase1_epochs": cfg.phase1_epochs,
            "phase2_epochs": cfg.phase2_epochs,
        },
        tmp,
    )
    tmp.replace(path)  # atomic: a crash mid-write never corrupts last.pt


METRICS_FIELDS = ["epoch", "phase", "lr", "train_loss", "val_loss", "val_auroc_mean"]


def _init_metrics_csv(path: Path, label_names: List[str], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        return
    fields = METRICS_FIELDS + [f"val_auroc_{l}" for l in label_names]
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(fields)


def _append_metrics(path: Path, row: Dict, label_names: List[str]) -> None:
    fields = METRICS_FIELDS + [f"val_auroc_{l}" for l in label_names]
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_phase(
    *, model, loaders, criterion, device, cfg, phase_name, epochs, lr,
    start_epoch, best_auroc, scaler, amp, label_names,
    phase_offset=0, resume_opt_state=None,
):
    """Run one training phase.

    phase_offset       : epochs of THIS phase already completed (for --resume).
    resume_opt_state   : optimizer.state_dict() to reload (only when resuming
                         mid-phase; ignored across a phase boundary since the
                         param groups differ between frozen/unfrozen).
    start_epoch        : global epoch count before this phase's first epoch.
    """
    import torch

    optimizer = torch.optim.Adam(
        trainable_parameters(model), lr=lr, weight_decay=cfg.weight_decay
    )
    if resume_opt_state is not None and phase_offset > 0:
        try:
            optimizer.load_state_dict(resume_opt_state)
            logger.info("  restored optimizer state for %s", phase_name)
        except ValueError as e:
            logger.warning("  could not restore optimizer state (%s) - continuing fresh", e)

    trn, tot = count_parameters(model)
    logger.info(
        "=== %s | epochs=%d (from %d) | lr=%.1e | trainable params=%s / %s ===",
        phase_name, epochs, phase_offset + 1, lr, f"{trn:,}", f"{tot:,}",
    )

    for e in range(phase_offset + 1, epochs + 1):
        epoch = start_epoch + e
        t0 = time.time()
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler, amp,
            desc=f"{phase_name} e{e}/{epochs} [train]",
        )
        val = evaluate_split(
            model, loaders["val"], criterion, device, label_names, amp,
            desc=f"{phase_name} e{e}/{epochs} [val]",
        )
        dt = time.time() - t0
        va = val["auroc"]
        per_label = " ".join(f"{l}={va[l]:.4f}" for l in label_names)
        logger.info(
            "epoch %02d | %s | train_loss=%.4f val_loss=%.4f | val_AUROC mean=%.4f (%s) | %.0fs",
            epoch, phase_name, train_loss, val["loss"], va["mean"], per_label, dt,
        )

        row = {
            "epoch": epoch, "phase": phase_name, "lr": lr,
            "train_loss": round(train_loss, 6), "val_loss": round(val["loss"], 6),
            "val_auroc_mean": round(va["mean"], 6),
            **{f"val_auroc_{l}": round(va[l], 6) for l in label_names},
        }
        _append_metrics(C.METRICS_CSV, row, label_names)

        # best.pt: whenever mean val AUROC improves, any epoch, any phase
        if np.isfinite(va["mean"]) and va["mean"] > best_auroc:
            best_auroc = va["mean"]
            _save(C.CHECKPOINT_BEST, model=model, optimizer=optimizer, scaler=scaler,
                  cfg=cfg, global_epoch=epoch, phase=phase_name, phase_offset=e,
                  best_auroc=best_auroc, val_auroc=va)
            logger.info("  ^ new best mean val AUROC=%.4f -> saved %s", best_auroc, C.CHECKPOINT_BEST.name)

        # last.pt: every epoch, for --resume
        _save(C.CHECKPOINT_LAST, model=model, optimizer=optimizer, scaler=scaler,
              cfg=cfg, global_epoch=epoch, phase=phase_name, phase_offset=e,
              best_auroc=best_auroc, val_auroc=va)

    return start_epoch + epochs, best_auroc


def main(cfg: "C.RunConfig", resume: bool = False) -> None:
    import torch

    set_seed(cfg.seed)
    device, dev_name, vram = describe_device()

    loaders = make_dataloaders(cfg)
    for req in ("train", "val"):
        if loaders.get(req) is None:
            raise RuntimeError(f"No data for split '{req}'. Run data exploration first.")

    # pos_weight from TRAIN prevalence -> counteract heavy negative imbalance
    train_df = loaders["_subsets"]["train"]
    pos = train_df[cfg.target_labels].sum().to_numpy()
    neg = len(train_df) - pos
    pos_weight = torch.tensor(
        np.where(pos > 0, neg / np.maximum(pos, 1), 1.0), dtype=torch.float32, device=device
    )
    logger.info("pos_weight (per label %s): %s", cfg.target_labels, pos_weight.tolist())

    model = build_model(num_classes=cfg.num_classes, pretrained=True).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    label_names = list(cfg.target_labels)

    # ---- resume bookkeeping ----
    best_auroc = -1.0
    done_global = 0
    resume_opt_state = None
    if resume:
        if not C.CHECKPOINT_LAST.exists():
            raise SystemExit(f"--resume: no checkpoint at {C.CHECKPOINT_LAST}")
        ckpt = torch.load(C.CHECKPOINT_LAST, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        best_auroc = ckpt.get("best_auroc", -1.0)
        done_global = int(ckpt.get("global_epoch", 0))
        resume_opt_state = ckpt.get("optimizer_state")
        logger.info(
            "RESUME from %s: %d/%d epochs done, best mean val AUROC so far=%.4f",
            C.CHECKPOINT_LAST.name, done_global, cfg.phase1_epochs + cfg.phase2_epochs, best_auroc,
        )

    _init_metrics_csv(C.METRICS_CSV, label_names, append=resume)
    logger.info("Metrics -> %s", C.METRICS_CSV)

    p1_done = min(done_global, cfg.phase1_epochs)
    p2_done = max(0, done_global - cfg.phase1_epochs)
    epoch = p1_done  # global epoch counter (== epochs completed so far in phase 1 terms)

    # ---- Phase 1: frozen backbone (final dense block + head trainable) ----
    freeze_backbone(model, train_last_block=True)
    if p1_done < cfg.phase1_epochs:
        epoch, best_auroc = run_phase(
            model=model, loaders=loaders, criterion=criterion, device=device, cfg=cfg,
            phase_name="phase1-frozen", epochs=cfg.phase1_epochs, lr=cfg.phase1_lr,
            start_epoch=0, best_auroc=best_auroc, scaler=scaler, amp=amp,
            label_names=label_names,
            phase_offset=p1_done, resume_opt_state=resume_opt_state,
        )
        resume_opt_state = None  # consumed
    else:
        logger.info("Phase 1 already complete (%d epochs) - skipping.", cfg.phase1_epochs)
        epoch = cfg.phase1_epochs

    # ---- Phase 2: full fine-tune ----
    if cfg.phase2_epochs > 0 and p2_done < cfg.phase2_epochs:
        unfreeze_all(model)
        epoch, best_auroc = run_phase(
            model=model, loaders=loaders, criterion=criterion, device=device, cfg=cfg,
            phase_name="phase2-unfrozen", epochs=cfg.phase2_epochs, lr=cfg.phase2_lr,
            start_epoch=cfg.phase1_epochs, best_auroc=best_auroc, scaler=scaler, amp=amp,
            label_names=label_names,
            phase_offset=p2_done, resume_opt_state=resume_opt_state,
        )
    elif cfg.phase2_epochs > 0:
        logger.info("Phase 2 already complete (%d epochs) - skipping.", cfg.phase2_epochs)

    logger.info("Done. Best mean val AUROC = %.4f | best ckpt: %s | last ckpt: %s",
                best_auroc, C.CHECKPOINT_BEST, C.CHECKPOINT_LAST)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DenseNet121 Cardiomegaly/Effusion classifier")
    p.add_argument("--quick_test", action="store_true",
                   help="500 train / 100 val images, 1 epoch, phase 1 only")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    p.add_argument("--resume", action="store_true",
                   help=f"resume from {C.CHECKPOINT_LAST.name} (saved every epoch)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = C.RunConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_amp:
        cfg.amp = False
    if args.quick_test:
        cfg.apply_quick_test()
    main(cfg, resume=args.resume)
