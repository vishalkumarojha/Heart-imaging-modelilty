"""Phase 3 data exploration / sanity report. Run BEFORE any model code.

    python -m src.mri_explore

Prints: patient count, file-naming check, diagnosis distribution, ED/ES frame
numbers, volume-shape + voxel-spacing variability, intensity ranges, a preview
of the chosen [N_SLICES, 3, HW, HW] preprocessing, the stratified split it
creates, and per-volume normalisation stats. Writes
data/processed/mri/{split_index.csv, norm_stats.json}.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from . import mri_config as MC
from .utils import setup_logging

logger = setup_logging()


def parse_info_cfg(patient_dir) -> dict:
    d = {}
    for line in (patient_dir / "Info.cfg").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def list_patients():
    return sorted(p for p in MC.ACDC_RAW_DIR.glob("patient*") if p.is_dir())


def build_records() -> list:
    """One dict per patient: id, group, label, ed/es frame numbers, file paths."""
    recs = []
    for pdir in list_patients():
        pid = pdir.name
        cfg = parse_info_cfg(pdir)
        ed, es = int(cfg["ED"]), int(cfg["ES"])
        rec = {
            "patient_id": pid,
            "group": cfg["Group"],
            "label": MC.GROUP_TO_IDX[cfg["Group"]],
            "ed_frame": ed,
            "es_frame": es,
            "ed_path": str(pdir / f"{pid}_frame{ed:02d}.nii.gz"),
            "es_path": str(pdir / f"{pid}_frame{es:02d}.nii.gz"),
            "dir": str(pdir),
        }
        recs.append(rec)
    return recs


def stratified_split(recs: list, fractions=MC.SPLIT_FRACTIONS, seed=MC.SEED) -> None:
    """Assign rec['split'] in place — stratified by group, one patient per row."""
    rng = np.random.default_rng(seed)
    by_group: dict = {}
    for r in recs:
        by_group.setdefault(r["group"], []).append(r)
    for group, group_recs in by_group.items():
        idx = np.arange(len(group_recs))
        rng.shuffle(idx)
        n = len(group_recs)
        n_tr = round(fractions[0] * n)
        n_va = round(fractions[1] * n)
        for k, j in enumerate(idx):
            group_recs[j]["split"] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")


def load_volume(path: str) -> np.ndarray:
    import nibabel as nib

    return np.asanyarray(nib.load(path).dataobj, dtype=np.float32)


def robust_normalize(vol: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol, [MC.CLIP_PCT_LOW, MC.CLIP_PCT_HIGH])
    vol = np.clip(vol, lo, hi)
    m, s = float(vol.mean()), float(vol.std())
    return (vol - m) / (s + 1e-6)


def preprocess_preview(ed_path: str, es_path: str) -> np.ndarray:
    """Emulate the planned loader: -> [N_SLICES, 3, HW, HW] (ED, ES, ED-ES)."""
    from scipy.ndimage import zoom

    ed = robust_normalize(load_volume(ed_path))   # (X, Y, Z)
    es = robust_normalize(load_volume(es_path))
    z = ed.shape[2]
    sl = np.round(np.linspace(0, z - 1, MC.N_SLICES)).astype(int)
    ed, es = ed[:, :, sl], es[:, :, sl]           # (X, Y, N)
    out = np.zeros((MC.N_SLICES, 3, MC.SLICE_HW, MC.SLICE_HW), np.float32)
    for i in range(MC.N_SLICES):
        a, b = ed[:, :, i], es[:, :, i]
        fy, fx = MC.SLICE_HW / a.shape[0], MC.SLICE_HW / a.shape[1]
        a = zoom(a, (fy, fx), order=1)
        b = zoom(b, (fy, fx), order=1)
        out[i, 0], out[i, 1], out[i, 2] = a, b, a - b
    return out


def main() -> None:
    logger.info("=" * 70)
    logger.info("PHASE 3 (ACDC cardiac MRI) DATA EXPLORATION")
    logger.info("=" * 70)

    recs = build_records()
    logger.info("patient folders: %d", len(recs))

    # ---- file-naming check ----
    import os

    missing = []
    for r in recs:
        for key in ("ed_path", "es_path"):
            if not os.path.exists(r[key]):
                missing.append(r[key])
            gt = r[key].replace(".nii.gz", "_gt.nii.gz")
            if not os.path.exists(gt):
                missing.append(gt)
    logger.info("naming pattern: patientXXX_frame{ED|ES:02d}.nii.gz (+ _gt) — missing files: %s",
                missing if missing else "NONE")

    # ---- diagnosis distribution ----
    logger.info("-" * 70)
    dist = Counter(r["group"] for r in recs)
    logger.info("Diagnosis distribution (index: name = count):")
    for name in MC.CLASS_NAMES:
        logger.info("  %d: %-5s = %d", MC.GROUP_TO_IDX[name], name, dist[name])
    logger.info("  -> %s", "BALANCED (challenge dataset)" if len(set(dist.values())) == 1
                else "IMBALANCED - enable class weights")

    eds = Counter(r["ed_frame"] for r in recs)
    ess = [r["es_frame"] for r in recs]
    logger.info("ED frame #: %s   |   ES frame #: min=%d max=%d median=%d",
                dict(eds), min(ess), max(ess), int(np.median(ess)))

    # ---- shape / spacing / intensity variability ----
    logger.info("-" * 70)
    import nibabel as nib

    shapes, spacings, inten = [], [], []
    for r in recs:
        img = nib.load(r["ed_path"])
        shapes.append(img.shape)
        spacings.append(tuple(round(float(x), 2) for x in img.header.get_zooms()[:3]))
        arr = np.asanyarray(img.dataobj, dtype=np.float32)
        inten.append((arr.min(), arr.max(), arr.mean(), np.percentile(arr, 99)))
    xs = np.array([s[0] for s in shapes]); ys = np.array([s[1] for s in shapes])
    zs = np.array([s[2] for s in shapes])
    logger.info("ED volume shape (X, Y, Z=slices):")
    logger.info("  X: %d-%d (median %d)   Y: %d-%d (median %d)   Z: %d-%d (median %d)",
                xs.min(), xs.max(), int(np.median(xs)), ys.min(), ys.max(), int(np.median(ys)),
                zs.min(), zs.max(), int(np.median(zs)))
    logger.info("  most common shapes: %s",
                [f"{s}x{c}" for s, c in Counter(shapes).most_common(5)])
    inplane = Counter(sp[0] for sp in spacings)
    thru = Counter(sp[2] for sp in spacings)
    logger.info("  in-plane spacing (mm): %s", dict(sorted(inplane.items())))
    logger.info("  through-plane spacing (mm): %s", dict(sorted(thru.items())))
    im = np.array(inten)
    logger.info("  intensity: min %.0f..%.0f | max %.0f..%.0f | mean %.1f | p99 %.0f..%.0f "
                "(huge inter-patient range -> per-volume normalisation)",
                im[:, 0].min(), im[:, 0].max(), im[:, 1].min(), im[:, 1].max(),
                im[:, 2].mean(), im[:, 3].min(), im[:, 3].max())

    # ---- preprocessing preview ----
    logger.info("-" * 70)
    prev = preprocess_preview(recs[0]["ed_path"], recs[0]["es_path"])
    logger.info("preprocessing preview (%s): input tensor shape %s, dtype %s",
                recs[0]["patient_id"], tuple(prev.shape), prev.dtype)
    logger.info("  per-channel mean/std after robust-norm+resize: "
                "ED %.3f/%.3f  ES %.3f/%.3f  (ED-ES) %.3f/%.3f",
                prev[:, 0].mean(), prev[:, 0].std(), prev[:, 1].mean(), prev[:, 1].std(),
                prev[:, 2].mean(), prev[:, 2].std())

    # ---- split ----
    logger.info("-" * 70)
    stratified_split(recs)
    logger.info("Stratified split (seed %d, %s):", MC.SEED, MC.SPLIT_FRACTIONS)
    logger.info("  %-6s %6s | %s", "split", "n", "per-class")
    for s in ("train", "val", "test"):
        sub = [r for r in recs if r["split"] == s]
        cc = Counter(r["group"] for r in sub)
        logger.info("  %-6s %6d | %s", s, len(sub),
                    " ".join(f"{name}={cc[name]}" for name in MC.CLASS_NAMES))

    import pandas as pd

    cols = ["patient_id", "group", "label", "split", "ed_frame", "es_frame",
            "ed_path", "es_path", "dir"]
    pd.DataFrame(recs)[cols].to_csv(MC.SPLIT_CSV, index=False)
    logger.info("  wrote %s", MC.SPLIT_CSV)

    # ---- normalisation stats from TRAIN subset ----
    logger.info("-" * 70)
    train_recs = [r for r in recs if r["split"] == "train"]
    ch = np.zeros((3, 2))  # sum, sumsq per channel
    npx = 0
    for r in train_recs:
        t = preprocess_preview(r["ed_path"], r["es_path"])   # [N,3,HW,HW]
        for c in range(3):
            ch[c, 0] += t[:, c].sum()
            ch[c, 1] += (t[:, c] ** 2).sum()
        npx += t[:, 0].size
    mean = (ch[:, 0] / npx)
    std = np.sqrt(np.clip(ch[:, 1] / npx - mean ** 2, 1e-8, None))
    stats = {"mean": [round(float(x), 5) for x in mean],
             "std": [round(float(x), 5) for x in std],
             "n_patients": len(train_recs), "note": "post robust-per-volume-norm residual stats"}
    MC.NORM_STATS_JSON.write_text(json.dumps(stats, indent=2))
    logger.info("residual channel mean=%s std=%s (from %d train patients) -> %s",
                stats["mean"], stats["std"], len(train_recs), MC.NORM_STATS_JSON)

    # ---- device ----
    logger.info("-" * 70)
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            logger.info("CUDA OK: %s | %.2f GB | cc %d.%d",
                        torch.cuda.get_device_name(0), p.total_memory / 1024**3, p.major, p.minor)
        else:
            logger.warning("CUDA NOT available")
    except ModuleNotFoundError:
        logger.warning("torch not importable")


if __name__ == "__main__":
    main()
