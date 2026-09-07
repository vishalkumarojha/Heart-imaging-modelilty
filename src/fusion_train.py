"""Phase 4 — train & validate the fusion pathway in SINGLE-MODALITY-PRESENT mode.

    python -m src.fusion_train                 # precompute (if needed) -> train -> validate
    python -m src.fusion_train --quick_test    # 64 samples/split, 1 epoch
    python -m src.fusion_train --precompute_only
    python -m src.fusion_train --skip_precompute --validate_only

================================ IMPORTANT ================================
 NIH ChestX-ray14 / EchoNet-Dynamic / ACDC have NO shared patients. There is
 no real tri-modal data. This script ONLY ever presents one real modality at a
 time (its own real test set; the other two replaced by the learned
 missing-modality token) and reports each modality's own real metric against
 its Phase 1-3 standalone baseline. Multi-modal combination is done ONLY in
 fusion_demo.py and is explicitly SYNTHETIC.
=========================================================================
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import fusion_config as FC
from .fusion_model import FusionModel, count_parameters
from .load_encoders import load_all_encoders
from .utils import binary_rates_from_confusion, multilabel_auroc, set_seed, setup_logging

logger = setup_logging()


# --------------------------------------------------------------------------- #
# 1. Embedding precompute  (frozen encoders -> cached [N,1024] vectors)
# --------------------------------------------------------------------------- #
def _xray_subset(split: str, cap):
    from . import config as XC
    from .dataset import MultiLabelImageDataset, build_dataset_frame, build_transforms

    frame, _ = build_dataset_frame(target_labels=XC.TARGET_LABELS)
    sub = frame[frame["split"] == split].reset_index(drop=True)
    if cap and len(sub) > cap:
        sub = sub.sample(cap, random_state=FC.SEED).reset_index(drop=True)
    ds = MultiLabelImageDataset(sub, ["Cardiomegaly", "Effusion"],
                                transform=build_transforms(False, 224), path_column="path")
    return ds, sub["Image Index"].tolist(), 64


def _echo_subset(split: str, cap):
    from . import echo_config as EC
    from .echo_dataset import EchoVideoDataset, build_echo_frame

    cfg = EC.EchoRunConfig()
    sub = build_echo_frame()
    sub = sub[sub["split"] == split].reset_index(drop=True)
    if cap and len(sub) > cap:
        sub = sub.sample(cap, random_state=FC.SEED).reset_index(drop=True)
    return EchoVideoDataset(sub, cfg, train=False), sub["FileName"].tolist(), 16


def _mri_subset(split: str, cap):
    from . import mri_config as MC
    from .mri_dataset import MriVolumeDataset, build_mri_frame

    cfg = MC.MriRunConfig()
    sub = build_mri_frame()
    sub = sub[sub["split"] == split].reset_index(drop=True)
    if cap and len(sub) > cap:
        sub = sub.sample(cap, random_state=FC.SEED).reset_index(drop=True)
    return MriVolumeDataset(sub, cfg, train=False), sub["patient_id"].tolist(), 8


_SUBSET = {"xray": _xray_subset, "echo": _echo_subset, "mri": _mri_subset}
# per-split caps: TEST is ALWAYS full (None) so validation numbers are honest
_CAPS = {
    "xray": {"train": FC.XRAY_TRAIN_CAP, "val": FC.XRAY_VAL_CAP, "test": None},
    "echo": {"train": None, "val": None, "test": None},
    "mri": {"train": None, "val": None, "test": None},
}


def _run_encoder(enc, ds, device, batch: int):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    embs, labels = [], []
    enc.eval()
    # fp32 on purpose: the DenseNet (X-ray) can overflow to inf in fp16 given its
    # large pre-classifier activations. One-time pass, correctness > speed.
    with torch.no_grad():
        for x, y in tqdm(dl, desc="  encode", leave=False):
            x = x.to(device, non_blocking=True)
            e = enc(x)
            embs.append(e.float().cpu().numpy())
            labels.append(y.numpy())
    emb = np.concatenate(embs)
    if not np.isfinite(emb).all():
        n = int((~np.isfinite(emb)).any(1).sum())
        logger.warning("  %d/%d embedding rows had non-finite values -> zeroed", n, len(emb))
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    return emb, np.concatenate(labels)


def precompute_embeddings(device, quick: bool = False, force: bool = False) -> None:
    logger.info("=" * 74)
    logger.info("EMBEDDING PRECOMPUTE  (frozen Phase 1-3 encoders -> %s)", FC.EMB_CACHE_DIR)
    logger.info("  X-ray TRAIN capped at %s, VAL at %s; every TEST split is FULL/uncapped.",
                FC.XRAY_TRAIN_CAP, FC.XRAY_VAL_CAP)
    logger.info("=" * 74)
    bundles = load_all_encoders(device, freeze=True)

    for mod in FC.MODALITIES:
        for split in ("train", "val", "test"):
            out = FC.EMB_CACHE_DIR / f"{mod}_{split}.npz"
            if out.exists() and not force and not quick:
                logger.info("  [%s/%s] cached (%s)", mod, split, out.name)
                continue
            cap = 64 if quick else _CAPS[mod][split]
            ds, ids, batch = _SUBSET[mod](split, cap)
            t0 = time.time()
            emb, label = _run_encoder(bundles[mod].encoder, ds, device, batch)
            np.savez(out, emb=emb.astype(np.float32), label=label,
                     id=np.array(ids[: len(emb)], dtype=object))
            logger.info("  [%s/%s] %5d samples -> %s  (%.0fs)  emb%s label%s",
                        mod, split, len(emb), out.name, time.time() - t0,
                        emb.shape, label.shape)


# --------------------------------------------------------------------------- #
# 2. Cached-embedding datasets
# --------------------------------------------------------------------------- #
class EmbCache:
    def __init__(self, mod: str, split: str) -> None:
        import torch

        d = np.load(FC.EMB_CACHE_DIR / f"{mod}_{split}.npz", allow_pickle=True)
        self.mod = mod
        self.emb = torch.from_numpy(d["emb"]).float()
        lab = d["label"]
        self.label = torch.from_numpy(lab).float() if lab.ndim == 2 else torch.from_numpy(lab).long()
        self.ids = list(d["id"])

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, i):
        return self.emb[i], self.label[i]


# --------------------------------------------------------------------------- #
# 3. Metrics  (per modality, matching its Phase 1-3 report)
# --------------------------------------------------------------------------- #
def _sanitize(logits: np.ndarray) -> np.ndarray:
    if not np.isfinite(logits).all():
        bad = int((~np.isfinite(logits)).any(1).sum())
        logger.warning("  %d/%d logit rows non-finite -> zeroed for metrics", bad, len(logits))
        logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    return logits


def _xray_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    logits = _sanitize(logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    classes = FC.TASK_SPECS["xray"]["classes"]
    auroc = multilabel_auroc(y_true, probs, classes)
    pred = (probs >= 0.5).astype(int)
    rows = {}
    for i, c in enumerate(classes):
        r = binary_rates_from_confusion(y_true[:, i], pred[:, i])
        rows[c] = {"auroc": auroc[c], "sensitivity": r["sensitivity"],
                   "specificity": r["specificity"], "precision": r["precision"], "f1": r["f1"]}
    return {"primary": auroc["mean"], "per_class": rows, "mean_auroc": auroc["mean"]}


def _multiclass_metrics(y_true: np.ndarray, logits: np.ndarray, classes: List[str]) -> dict:
    logits = _sanitize(logits)
    z = logits - logits.max(1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y_true = y_true.astype(int)
    auroc = multilabel_auroc(np.eye(len(classes))[y_true], probs, classes)
    pred = probs.argmax(1)
    k = len(classes)
    cm = np.zeros((k, k), int)
    for t, p in zip(y_true, pred):
        cm[t, p] += 1
    recalls = [cm[c, c] / cm[c].sum() for c in range(k) if cm[c].sum()]
    f1s = []
    for c in range(k):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c].sum() - cm[c, c]
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {"primary": auroc["mean"], "macro_auroc": auroc["mean"],
            "per_class_auroc": {c: auroc[c] for c in classes},
            "per_class_f1": dict(zip(classes, f1s)),
            "accuracy": float((pred == y_true).mean()),
            "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
            "macro_f1": float(np.mean(f1s)), "confusion": cm.tolist()}


def evaluate_modality(model, cache: EmbCache, device) -> dict:
    """Run one modality's cached embeddings through fusion with ONLY that
    modality present, return its full metric dict."""
    import torch

    mod = cache.mod
    mi = FC.MODALITIES.index(mod)
    model.eval()
    logits_all = []
    with torch.no_grad():
        for s in range(0, len(cache), 4096):
            e = cache.emb[s:s + 4096].to(device)
            embeds = [None, None, None]
            embeds[mi] = e
            mask = model.mask_for([mod], e.shape[0], device)
            out, _ = model(embeds, mask)
            logits_all.append(out[mod].float().cpu().numpy())
    logits = np.concatenate(logits_all)
    y = cache.label.numpy()
    if FC.TASK_SPECS[mod]["type"] == "multilabel":
        return _xray_metrics(y, logits)
    return _multiclass_metrics(y, logits, FC.TASK_SPECS[mod]["classes"])


# --------------------------------------------------------------------------- #
# 4. Train
# --------------------------------------------------------------------------- #
def train(cfg: "FC.FusionRunConfig", device) -> None:
    import torch
    from torch.utils.data import DataLoader

    set_seed(cfg.seed)
    train_caches = {m: EmbCache(m, "train") for m in FC.MODALITIES}
    val_caches = {m: EmbCache(m, "val") for m in FC.MODALITIES}
    loaders = {
        m: DataLoader(c, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
        for m, c in train_caches.items()
    }
    logger.info("train sizes: %s | val sizes: %s",
                {m: len(c) for m, c in train_caches.items()},
                {m: len(c) for m, c in val_caches.items()})

    model = FusionModel().to(device)
    trn, tot = count_parameters(model)
    logger.info("FusionModel trainable %s / %s params (encoders are frozen & external)",
                f"{trn:,}", f"{tot:,}")

    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    fields = ["epoch"] + [f"train_loss_{m}" for m in FC.MODALITIES] \
        + [f"val_{FC.TASK_SPECS[m]['primary']}_{m}" for m in FC.MODALITIES] + ["val_combined"]
    FC.METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with FC.METRICS_CSV.open("w", newline="") as f:
        csv.writer(f).writerow(fields)

    best_combined = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss = {m: 0.0 for m in FC.MODALITIES}
        ep_n = {m: 0 for m in FC.MODALITIES}
        for mod in FC.MODALITIES:
            mi = FC.MODALITIES.index(mod)
            for emb, label in loaders[mod]:
                emb = emb.to(device)
                label = label.to(device)
                embeds = [None, None, None]
                embeds[mi] = emb
                mask = model.mask_for([mod], emb.shape[0], device)
                out, _ = model(embeds, mask)
                if FC.TASK_SPECS[mod]["type"] == "multilabel":
                    loss = bce(out[mod], label)
                else:
                    loss = ce(out[mod], label)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ep_loss[mod] += loss.item() * emb.shape[0]
                ep_n[mod] += emb.shape[0]

        val = {m: evaluate_modality(model, val_caches[m], device)["primary"] for m in FC.MODALITIES}
        combined = float(np.mean(list(val.values())))
        row = {"epoch": epoch, "val_combined": round(combined, 6)}
        for m in FC.MODALITIES:
            row[f"train_loss_{m}"] = round(ep_loss[m] / max(ep_n[m], 1), 6)
            row[f"val_{FC.TASK_SPECS[m]['primary']}_{m}"] = round(val[m], 6)
        with FC.METRICS_CSV.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        logger.info("epoch %02d | loss x=%.3f e=%.3f m=%.3f | val primary x=%.3f e=%.3f m=%.3f "
                    "| combined=%.4f | %.1fs",
                    epoch, row["train_loss_xray"], row["train_loss_echo"], row["train_loss_mri"],
                    val["xray"], val["echo"], val["mri"], combined, time.time() - t0)

        _save(model, opt, epoch, combined, FC.CKPT_LAST)
        if combined > best_combined:
            best_combined = combined
            _save(model, opt, epoch, combined, FC.CKPT_BEST)
            logger.info("  ^ new best combined val = %.4f -> %s", best_combined, FC.CKPT_BEST.name)

    logger.info("Done. best combined val primary = %.4f", best_combined)


def _save(model, opt, epoch, combined, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"epoch": epoch, "combined_val": combined,
                "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                "scheme": FC.UNIFIED_SCHEME, "modalities": FC.MODALITIES,
                "task_specs": FC.TASK_SPECS,
                "NOTE": "single-modality-present training; no real tri-modal data"}, tmp)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# 5. Single-modality-present validation on the REAL test sets
# --------------------------------------------------------------------------- #
def validate_single_modality(device, checkpoint: Path = FC.CKPT_BEST) -> None:
    import torch

    from .fusion_model import FusionModel as _FM

    ck = torch.load(checkpoint, map_location=device)
    model = _FM().to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    logger.info("=" * 74)
    logger.info("SINGLE-MODALITY-PRESENT VALIDATION on the REAL Phase 1-3 TEST sets")
    logger.info("  (each modality real; other two = learned missing-modality token)")
    logger.info("  checkpoint: %s (epoch %s)", checkpoint.name, ck.get("epoch"))
    logger.info("=" * 74)

    summary_rows = []
    for mod in FC.MODALITIES:
        cache = EmbCache(mod, "test")
        m = evaluate_modality(model, cache, device)
        spec = FC.TASK_SPECS[mod]
        base = spec["standalone_test"]
        delta = m["primary"] - base
        logger.info("-" * 74)
        logger.info("[%s]  %s  |  N=%d  |  original task: %s", mod.upper(),
                    spec["classes"], len(cache), spec["type"])
        logger.info("  fusion-pathway %s = %.4f   vs   Phase %d standalone test = %.3f   (Δ %+.3f)",
                    spec["primary"], m["primary"], FC.MODALITIES.index(mod) + 1, base, delta)
        if spec["type"] == "multilabel":
            for c, r in m["per_class"].items():
                logger.info("    %-14s AUROC=%.4f  sens=%.3f spec=%.3f prec=%.3f F1=%.3f",
                            c, r["auroc"], r["sensitivity"], r["specificity"], r["precision"], r["f1"])
        else:
            for c in spec["classes"]:
                logger.info("    %-16s AUROC=%.4f  F1=%.3f", c,
                            m["per_class_auroc"][c], m["per_class_f1"][c])
            logger.info("    accuracy=%.3f  balanced_accuracy=%.3f  macro_F1=%.3f",
                        m["accuracy"], m["balanced_accuracy"], m["macro_f1"])
            logger.info("    confusion (rows=true, cols=pred) %s:", spec["classes"])
            for c, r in zip(spec["classes"], m["confusion"]):
                logger.info("      %-16s %s", c, r)
        summary_rows.append({
            "modality": mod, "n_test": len(cache), "primary_metric": spec["primary"],
            "fusion_pathway": round(m["primary"], 4), "standalone_baseline": base,
            "delta": round(delta, 4),
            "accuracy": round(m.get("accuracy", float("nan")), 4)
            if spec["type"] == "multiclass" else "",
        })

    with FC.SINGLE_MODALITY_RESULTS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    logger.info("=" * 74)
    logger.info("wrote %s", FC.SINGLE_MODALITY_RESULTS_CSV)
    logger.info("READ: these are the ONLY real fusion numbers. 'fusion_pathway' near "
                "'standalone_baseline' == the fusion block preserves each encoder's signal.")
    logger.info("No multi-modal numbers exist — no shared patients across the 3 datasets.")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 fusion — single-modality-present train/validate")
    p.add_argument("--quick_test", action="store_true")
    p.add_argument("--precompute_only", action="store_true")
    p.add_argument("--skip_precompute", action="store_true")
    p.add_argument("--validate_only", action="store_true")
    p.add_argument("--force_precompute", action="store_true")
    return p.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FC.FusionRunConfig()
    if args.quick_test:
        cfg.apply_quick_test()

    if args.validate_only:
        validate_single_modality(device)
        return
    if not args.skip_precompute:
        precompute_embeddings(device, quick=args.quick_test, force=args.force_precompute)
    if args.precompute_only:
        return
    train(cfg, device)
    validate_single_modality(device, FC.CKPT_BEST)


if __name__ == "__main__":
    main()
