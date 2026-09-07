"""Phase 2 training: EF-category classifier (ResNet18 + BiLSTM).

Two-phase, mirroring Phase 1:
  Phase 1 : CNN frozen (BiLSTM + head train), lr 1e-4, 3 epochs
  Phase 2 : fully unfrozen,                   lr 1e-5, 12 epochs

Single-label 3-class -> CrossEntropyLoss with inverse-frequency class weights.
Checkpoint selection metric: macro one-vs-rest AUROC (accuracy is dominated by
the ~70% Normal class). last.pt every epoch; --resume restores everything.

    python -m src.echo_train
    python -m src.echo_train --resume
    python -m src.echo_train --quick_test
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import echo_config as EC
from .echo_dataset import compute_class_weights, make_echo_dataloaders
from .echo_model import (
    build_echo_model,
    count_parameters,
    freeze_cnn,
    load_encoder_from_checkpoint,  # noqa: F401  (re-exported for convenience)
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


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: List[str]) -> Dict:
    """probs: [N, 3] softmax. Returns macro/per-class OvR AUROC + accuracy + balanced acc."""
    y_true = np.asarray(y_true).astype(int)
    onehot = np.eye(len(class_names))[y_true]
    auroc = multilabel_auroc(onehot, probs, class_names)  # per-class OvR + "mean" == macro
    pred = probs.argmax(1)
    acc = float((pred == y_true).mean())
    recalls = []
    for c in range(len(class_names)):
        m = y_true == c
        if m.any():
            recalls.append(float((pred[m] == c).mean()))
    bal_acc = float(np.mean(recalls)) if recalls else float("nan")
    return {"macro_auroc": auroc["mean"], "per_class_auroc": auroc, "accuracy": acc,
            "balanced_accuracy": bal_acc}


# --------------------------------------------------------------------------- #
# Epoch loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp, frozen, desc):
    import torch
    from tqdm import tqdm

    model.train()
    if frozen:
        set_frozen_bn_eval(model)
    running, n = 0.0, 0
    for videos, labels in tqdm(loader, desc=desc, leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(videos)
            loss = criterion(logits, labels)
        if amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = videos.size(0)
        running += loss.item() * bs
        n += bs
    return running / max(n, 1)


def evaluate_split(model, loader, criterion, device, class_names, amp, desc="val"):
    import torch
    from tqdm import tqdm

    model.eval()
    running, n = 0.0, 0
    all_probs, all_true = [], []
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc=desc, leave=False):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(videos)
                loss = criterion(logits, labels)
            running += loss.item() * videos.size(0)
            n += videos.size(0)
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            all_true.append(labels.cpu().numpy())
    probs = np.concatenate(all_probs)
    true = np.concatenate(all_true)
    m = compute_metrics(true, probs, class_names)
    m["loss"] = running / max(n, 1)
    return m


# --------------------------------------------------------------------------- #
# Checkpoint / metrics IO
# --------------------------------------------------------------------------- #
def _save(path: Path, *, model, optimizer, scaler, cfg, global_epoch, phase, phase_offset,
          best_metric, val_metrics) -> None:
    import torch

    from .echo_dataset import load_norm_stats

    mean, std = load_norm_stats()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": global_epoch, "global_epoch": global_epoch,
            "phase": phase, "phase_offset": phase_offset,
            "best_metric": best_metric, "ckpt_metric": EC.CKPT_METRIC,
            "val_metrics": val_metrics,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "class_names": list(cfg.class_names),
            "frames_per_clip": cfg.frames_per_clip,
            "frame_size": cfg.frame_size,
            "d_echo": cfg.d_echo,
            "arch": "resnet18_bilstm",
            "norm_mean": list(mean), "norm_std": list(std),
            "phase1_epochs": cfg.phase1_epochs, "phase2_epochs": cfg.phase2_epochs,
        },
        tmp,
    )
    tmp.replace(path)


METRICS_FIELDS = ["epoch", "phase", "lr", "train_loss", "val_loss",
                  "val_macro_auroc", "val_accuracy", "val_balanced_accuracy"]


def _init_metrics_csv(path: Path, class_names: List[str], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        return
    fields = METRICS_FIELDS + [f"val_auroc_{c.replace(' ', '')}" for c in class_names]
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(fields)


def _append_metrics(path: Path, row: Dict, class_names: List[str]) -> None:
    fields = METRICS_FIELDS + [f"val_auroc_{c.replace(' ', '')}" for c in class_names]
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow(row)


# --------------------------------------------------------------------------- #
# Phase runner
# --------------------------------------------------------------------------- #
def run_phase(*, model, loaders, criterion, device, cfg, phase_name, epochs, lr, frozen,
              start_epoch, best_metric, scaler, amp, class_names,
              phase_offset=0, resume_opt_state=None):
    import torch

    optimizer = torch.optim.Adam(trainable_parameters(model), lr=lr, weight_decay=cfg.weight_decay)
    if resume_opt_state is not None and phase_offset > 0:
        try:
            optimizer.load_state_dict(resume_opt_state)
            logger.info("  restored optimizer state for %s", phase_name)
        except ValueError as e:
            logger.warning("  optimizer state not restored (%s)", e)

    trn, tot = count_parameters(model)
    logger.info("=== %s | epochs=%d (from %d) | lr=%.1e | trainable %s / %s ===",
                phase_name, epochs, phase_offset + 1, lr, f"{trn:,}", f"{tot:,}")

    for e in range(phase_offset + 1, epochs + 1):
        epoch = start_epoch + e
        t0 = time.time()
        tr_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device,
                                  scaler, amp, frozen, f"{phase_name} e{e}/{epochs} [train]")
        val = evaluate_split(model, loaders["val"], criterion, device, class_names, amp,
                             f"{phase_name} e{e}/{epochs} [val]")
        dt = time.time() - t0
        pc = val["per_class_auroc"]
        logger.info(
            "epoch %02d | %s | train_loss=%.4f val_loss=%.4f | macroAUROC=%.4f acc=%.4f balAcc=%.4f "
            "| per-class AUROC %s | %.0fs",
            epoch, phase_name, tr_loss, val["loss"], val["macro_auroc"], val["accuracy"],
            val["balanced_accuracy"],
            " ".join(f"{c}={pc[c]:.3f}" for c in class_names), dt,
        )
        row = {
            "epoch": epoch, "phase": phase_name, "lr": lr,
            "train_loss": round(tr_loss, 6), "val_loss": round(val["loss"], 6),
            "val_macro_auroc": round(val["macro_auroc"], 6),
            "val_accuracy": round(val["accuracy"], 6),
            "val_balanced_accuracy": round(val["balanced_accuracy"], 6),
            **{f"val_auroc_{c.replace(' ', '')}": round(pc[c], 6) for c in class_names},
        }
        _append_metrics(EC.METRICS_CSV, row, class_names)

        score = val["macro_auroc"]
        if np.isfinite(score) and score > best_metric:
            best_metric = score
            _save(EC.CKPT_BEST, model=model, optimizer=optimizer, scaler=scaler, cfg=cfg,
                  global_epoch=epoch, phase=phase_name, phase_offset=e, best_metric=best_metric,
                  val_metrics=val)
            logger.info("  ^ new best macro AUROC=%.4f -> saved %s", best_metric, EC.CKPT_BEST.name)
        _save(EC.CKPT_LAST, model=model, optimizer=optimizer, scaler=scaler, cfg=cfg,
              global_epoch=epoch, phase=phase_name, phase_offset=e, best_metric=best_metric,
              val_metrics=val)

    return start_epoch + epochs, best_metric


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(cfg: "EC.EchoRunConfig", resume: bool = False) -> None:
    import torch

    set_seed(cfg.seed)
    device = describe_device()

    loaders = make_echo_dataloaders(cfg)
    for req in ("train", "val"):
        if loaders.get(req) is None:
            raise RuntimeError(f"no data for split '{req}'")

    train_df = loaders["_subsets"]["train"]
    cw = compute_class_weights(train_df)
    class_weight = torch.tensor(cw, dtype=torch.float32, device=device)
    logger.info("class weights (%s): %s", cfg.class_names, [round(float(x), 3) for x in cw])

    model = build_echo_model(num_classes=cfg.num_classes, pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    class_names = list(cfg.class_names)

    best_metric = -1.0
    done_global = 0
    resume_opt_state = None
    if resume:
        if not EC.CKPT_LAST.exists():
            raise SystemExit(f"--resume: no checkpoint at {EC.CKPT_LAST}")
        ck = torch.load(EC.CKPT_LAST, map_location=device)
        model.load_state_dict(ck["model_state"])
        if ck.get("scaler_state") is not None:
            scaler.load_state_dict(ck["scaler_state"])
        best_metric = ck.get("best_metric", -1.0)
        done_global = int(ck.get("global_epoch", 0))
        resume_opt_state = ck.get("optimizer_state")
        logger.info("RESUME from %s: %d/%d epochs done, best macro AUROC=%.4f",
                    EC.CKPT_LAST.name, done_global, cfg.phase1_epochs + cfg.phase2_epochs, best_metric)

    _init_metrics_csv(EC.METRICS_CSV, class_names, append=resume)
    logger.info("Metrics -> %s", EC.METRICS_CSV)

    p1_done = min(done_global, cfg.phase1_epochs)
    p2_done = max(0, done_global - cfg.phase1_epochs)
    epoch = p1_done

    freeze_cnn(model)
    if p1_done < cfg.phase1_epochs:
        epoch, best_metric = run_phase(
            model=model, loaders=loaders, criterion=criterion, device=device, cfg=cfg,
            phase_name="phase1-frozen", epochs=cfg.phase1_epochs, lr=cfg.phase1_lr, frozen=True,
            start_epoch=0, best_metric=best_metric, scaler=scaler, amp=amp, class_names=class_names,
            phase_offset=p1_done, resume_opt_state=resume_opt_state)
        resume_opt_state = None
    else:
        logger.info("Phase 1 already complete - skipping.")
        epoch = cfg.phase1_epochs

    if cfg.phase2_epochs > 0 and p2_done < cfg.phase2_epochs:
        unfreeze_all(model)
        epoch, best_metric = run_phase(
            model=model, loaders=loaders, criterion=criterion, device=device, cfg=cfg,
            phase_name="phase2-unfrozen", epochs=cfg.phase2_epochs, lr=cfg.phase2_lr, frozen=False,
            start_epoch=cfg.phase1_epochs, best_metric=best_metric, scaler=scaler, amp=amp,
            class_names=class_names, phase_offset=p2_done, resume_opt_state=resume_opt_state)
    elif cfg.phase2_epochs > 0:
        logger.info("Phase 2 already complete - skipping.")

    logger.info("Done. Best macro AUROC = %.4f | best: %s | last: %s",
                best_metric, EC.CKPT_BEST, EC.CKPT_LAST)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EchoNet EF-category classifier (ResNet18+BiLSTM)")
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--frames", type=int, default=None, help="frames per clip")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = EC.EchoRunConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.frames is not None:
        cfg.frames_per_clip = args.frames
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_amp:
        cfg.amp = False
    if args.quick_test:
        cfg.apply_quick_test()
    main(cfg, resume=args.resume)
