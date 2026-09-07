"""Phase 2 data exploration / sanity report. Run BEFORE any model code.

    python -m src.echo_explore

Prints:
  * FileList.csv row count vs .avi files on disk (flags mismatches)
  * EF distribution + the 3-class bucket counts per official split
  * frame-count / FPS / resolution summary, decoded-vs-CSV frame-count check
  * per-channel mean/std computed from a TRAIN subset (cached to norm_stats.json)
  * device / CUDA / VRAM report
Also writes data/processed/echo/echo_index.csv (the master table).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import echo_config as EC
from .utils import setup_logging

logger = setup_logging()


def build_index() -> pd.DataFrame:
    df = pd.read_csv(EC.FILELIST_CSV)
    df.columns = [c.strip() for c in df.columns]
    df["path"] = df["FileName"].map(lambda n: str(EC.VIDEO_DIR / f"{n}.avi"))
    df["ef_bucket"] = df["EF"].astype(float).map(EC.bucket_ef)
    df["exists"] = df["path"].map(lambda p: Path(p).exists())
    return df


def _decode_all_frames(path: str):
    """Return (n_frames, (h, w), dtype_ok) decoding the whole clip with cv2."""
    import cv2

    cap = cv2.VideoCapture(path)
    n, shape = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if shape is None:
            shape = frame.shape
        n += 1
    cap.release()
    return n, shape


def compute_norm_stats(df: pd.DataFrame, n_videos: int = 250, frames_each: int = 8) -> dict:
    """Per-channel mean/std over a random TRAIN subset, scaled to [0,1]."""
    import cv2

    train = df[(df.Split == "TRAIN") & df.exists]
    sample = train.sample(min(n_videos, len(train)), random_state=EC.SEED)
    # Welford-ish accumulation in float64, per channel (RGB order from cv2 BGR->RGB)
    tot = np.zeros(3, np.float64)
    tot_sq = np.zeros(3, np.float64)
    count = 0
    t0 = time.time()
    for path in sample.path:
        cap = cv2.VideoCapture(path)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nf <= 0:
            cap.release()
            continue
        idx = np.linspace(0, nf - 1, min(frames_each, nf)).astype(int)
        want = set(int(i) for i in idx)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in want:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
                tot += rgb.reshape(-1, 3).sum(0)
                tot_sq += (rgb.reshape(-1, 3) ** 2).sum(0)
                count += rgb.shape[0] * rgb.shape[1]
            i += 1
        cap.release()
    mean = tot / count
    var = tot_sq / count - mean ** 2
    std = np.sqrt(np.clip(var, 1e-12, None))
    stats = {
        "mean": [round(float(x), 5) for x in mean],
        "std": [round(float(x), 5) for x in std],
        "n_videos": int(len(sample)),
        "n_pixels": int(count),
        "seconds": round(time.time() - t0, 1),
    }
    return stats


def device_report() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        logger.warning("torch not importable")
        return
    logger.info("torch %s | cv2 %s", torch.__version__, __import__("cv2").__version__)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        logger.info("CUDA OK: %s | %.2f GB VRAM | cc %d.%d",
                    torch.cuda.get_device_name(0), p.total_memory / 1024**3, p.major, p.minor)
    else:
        logger.warning("CUDA NOT available - CPU only")


def main() -> None:
    logger.info("=" * 70)
    logger.info("PHASE 2 (EchoNet-Dynamic) DATA EXPLORATION")
    logger.info("=" * 70)

    df = build_index()
    n_disk = len(list(EC.VIDEO_DIR.glob("*.avi")))
    logger.info("FileList.csv rows          : %d", len(df))
    logger.info(".avi files in Videos/      : %d", n_disk)
    logger.info("rows with file on disk     : %d", int(df.exists.sum()))
    missing = df[~df.exists]
    logger.info("rows WITHOUT a file        : %d %s", len(missing),
                "<-- MISMATCH" if len(missing) else "(clean)")
    if len(missing):
        logger.warning("  e.g. %s", missing.FileName.head(5).tolist())

    # ---- EF + buckets ----
    ef = df.EF.astype(float)
    logger.info("-" * 70)
    logger.info("EF (%%): min=%.1f  max=%.1f  mean=%.1f  median=%.1f  NaN=%d",
                ef.min(), ef.max(), ef.mean(), ef.median(), int(ef.isna().sum()))
    logger.info("Buckets: 0=Reduced(EF<%d)  1=Mildly(%d-%d)  2=Normal(EF>=%d)",
                EC.EF_BOUNDARY_REDUCED, EC.EF_BOUNDARY_REDUCED,
                EC.EF_BOUNDARY_NORMAL - 1, EC.EF_BOUNDARY_NORMAL)
    logger.info("%-8s %8s | %10s %10s %10s | %s", "split", "n", "Reduced", "Mildly", "Normal", "Normal:Reduced")
    for s in ["TRAIN", "VAL", "TEST"]:
        sub = df[df.Split == s]
        c = sub.ef_bucket.value_counts().reindex([0, 1, 2]).fillna(0).astype(int)
        pct = (c / max(len(sub), 1) * 100)
        ratio = c[2] / max(c[0], 1)
        logger.info("%-8s %8d | %4d (%4.1f%%) %4d (%4.1f%%) %4d (%4.1f%%) | %.1f : 1",
                    s, len(sub), c[0], pct[0], c[1], pct[1], c[2], pct[2], ratio)

    # ---- frame counts / fps / resolution ----
    logger.info("-" * 70)
    nf = df.NumberOfFrames.astype(int)
    logger.info("NumberOfFrames: min=%d  max=%d  mean=%.0f  median=%d  | videos <%d frames: %d",
                nf.min(), nf.max(), nf.mean(), nf.median(), EC.FRAMES_PER_CLIP,
                int((nf < EC.FRAMES_PER_CLIP).sum()))
    fps_mode = df.FPS.mode().iloc[0]
    logger.info("FPS: mode=%.0f (%d/%d videos); range %d-%d",
                fps_mode, int((df.FPS == fps_mode).sum()), len(df), int(df.FPS.min()), int(df.FPS.max()))
    bad_dims = df[(df.FrameHeight != EC.FRAME_SIZE) | (df.FrameWidth != EC.FRAME_SIZE)]
    logger.info("CSV rows with non-%dx%d dims: %d (decoded frames verified 112x112 below - loader resizes anyway)",
                EC.FRAME_SIZE, EC.FRAME_SIZE, len(bad_dims))

    # ---- decode check on a sample ----
    logger.info("-" * 70)
    logger.info("Decoding 8 sample videos (frame-count vs CSV)...")
    sample = df[df.exists].sample(8, random_state=EC.SEED)
    mism = 0
    for _, r in sample.iterrows():
        t0 = time.time()
        n, shape = _decode_all_frames(r.path)
        dt = (time.time() - t0) * 1000
        ok = n == int(r.NumberOfFrames)
        mism += (not ok)
        logger.info("  %s  decoded=%d csv=%d %s  shape=%s  %.0fms",
                    r.FileName, n, int(r.NumberOfFrames),
                    "OK" if ok else "MISMATCH", shape, dt)
    logger.info("frame-count mismatches in sample: %d/8", mism)

    # ---- normalization stats ----
    logger.info("-" * 70)
    logger.info("Computing channel mean/std from TRAIN subset (this is NOT ImageNet)...")
    stats = compute_norm_stats(df)
    EC.NORM_STATS_JSON.write_text(json.dumps(stats, indent=2))
    logger.info("  mean=%s  std=%s  (from %d videos, %s px, %.1fs)",
                stats["mean"], stats["std"], stats["n_videos"], f"{stats['n_pixels']:,}", stats["seconds"])
    logger.info("  cached -> %s", EC.NORM_STATS_JSON)

    # ---- cache index ----
    keep = ["FileName", "path", "EF", "ef_bucket", "Split", "FPS", "NumberOfFrames"]
    df[keep].to_csv(EC.INDEX_CACHE_CSV, index=False)
    logger.info("-" * 70)
    logger.info("Wrote master index -> %s (%d rows)", EC.INDEX_CACHE_CSV, len(df))

    logger.info("-" * 70)
    device_report()


if __name__ == "__main__":
    main()
