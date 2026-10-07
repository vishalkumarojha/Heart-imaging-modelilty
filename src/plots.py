"""Matplotlib figures for the calibration / threshold experiments.

matplotlib only (seaborn is not installed in this environment); the Agg backend
is selected before pyplot import so every figure can be written headlessly.

All functions are pure: they take data + an output path, write the file and
return the path. No figure is created outside a function, so a batch run cannot
leak state between plots.

    from src.plots import reliability_plot, curves_with_thresholds, calibration_bars
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402
from .calibration import reliability_bins  # noqa: E402

DPI = 150


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def reliability_plot(
    y_true: np.ndarray,
    probs: Dict[str, np.ndarray],
    title: str,
    path: Path,
    n_bins: int = C.CALIBRATION_BINS,
) -> Path:
    """Reliability diagram: one curve per entry of `probs` (e.g. raw vs calibrated)."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1,
            label="perfectly calibrated")
    for name, p in probs.items():
        b = reliability_bins(y_true, p, n_bins=n_bins)
        ok = b["count"] > 0
        ax.plot(b["mean_confidence"][ok], b["empirical_accuracy"][ok],
                marker="o", markersize=4, linewidth=1.5, label=name)
        # mass per bin (why the ECE is what it is)
        ax.bar(b["bin_centers"][ok], (b["count"][ok] / max(b["count"].sum(), 1)) * 0.15,
               width=(1.0 / n_bins) * 0.9, alpha=0.15, bottom=0)
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed frequency (bin)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def calibration_bars(
    values: Dict[str, Dict[str, Dict[str, float]]],
    path: Path,
    metrics: Sequence[str] = ("ece", "brier"),
    title: str = "Calibration quality (raw vs temperature-scaled)",
) -> Path:
    """Grouped bars: values = {metric: {label: {raw: x, calibrated: y}}}."""
    labels = list(next(iter(values.values())).keys())  # label names from first metric
    n_groups, n_series = len(labels), len(metrics)
    width = 0.8 / (n_series * 2)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(labels))
    styles = {
        ("ece", "raw"): ("#c0392b", "ECE raw"),
        ("ece", "calibrated"): ("#e67e22", "ECE calibrated"),
        ("brier", "raw"): ("#2980b9", "Brier raw"),
        ("brier", "calibrated"): ("#16a085", "Brier calibrated"),
    }
    i = 0
    for metric in metrics:
        for variant in ("raw", "calibrated"):
            series = [values[metric][lbl][variant] for lbl in labels]
            color, label = styles.get((metric, variant), (None, f"{metric} {variant}"))
            ax.bar(x + i * width, series, width * 0.95, color=color, label=label)
            i += 1
    ax.set_xticks(x + 0.8 / 2 - width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("lower is better")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
def curves_with_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Dict[str, float],
    title: str,
    path_roc: Path,
    path_pr: Path,
) -> Dict[str, Path]:
    """ROC + precision-recall curves with the operating point of every policy."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    prevalence = float(y_true.mean()) if y_true.size else float("nan")

    out: Dict[str, Path] = {}
    palette = plt.get_cmap("tab10")

    # ---- ROC ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(fpr, tpr, color="#2c3e50", linewidth=1.6, label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    for i, (name, tau) in enumerate(sorted(thresholds.items())):
        p = (y_prob >= tau).astype(int)
        tp = int(((p == 1) & (y_true == 1)).sum())
        fp = int(((p == 1) & (y_true == 0)).sum())
        fn = int(((p == 0) & (y_true == 1)).sum())
        tn = int(((p == 0) & (y_true == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        ax.scatter([1 - spec], [sens], s=28, color=palette(i % 10), zorder=5,
                   label=f"{name} (τ={tau:.3f})")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("sensitivity")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    out["roc"] = _save(fig, path_roc)

    # ---- PR ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(recall, precision, color="#8e44ad", linewidth=1.6, label="model")
    ax.axhline(prevalence, linestyle="--", color="#999999", linewidth=1,
               label=f"prevalence = {prevalence:.3f}")
    for i, (name, tau) in enumerate(sorted(thresholds.items())):
        p = (y_prob >= tau).astype(int)
        tp = int(((p == 1) & (y_true == 1)).sum())
        fp = int(((p == 1) & (y_true == 0)).sum())
        fn = int(((p == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        ax.scatter([rec], [prec], s=28, color=palette(i % 10), zorder=5,
                   label=f"{name} (τ={tau:.3f})")
    ax.set_xlabel("recall (sensitivity)")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    out["pr"] = _save(fig, path_pr)
    return out


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str,
    path: Path,
    markers: Optional[Dict[str, float]] = None,
) -> Path:
    """Precision / recall / F1 as a function of the threshold (why a policy picks τ)."""
    from .metrics import binary_metrics

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    taus = np.unique(y_prob)
    keep = np.linspace(0, len(taus) - 1, min(600, len(taus))).astype(int)
    taus = taus[keep]
    prec, rec, f1 = [], [], []
    for t in taus:
        m = binary_metrics(y_true, y_prob, float(t))
        prec.append(m["precision"])
        rec.append(m["recall"])
        f1.append(m["f1"])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(taus, prec, label="precision", linewidth=1.4)
    ax.plot(taus, rec, label="recall", linewidth=1.4)
    ax.plot(taus, f1, label="F1", linewidth=1.4, linestyle="--")
    for i, (name, t) in enumerate(sorted((markers or {}).items())):
        ax.axvline(t, color=plt.get_cmap("tab10")(i % 10), linewidth=1, alpha=0.7)
        ax.text(t, 1.02, name, rotation=90, fontsize=7,
                color=plt.get_cmap("tab10")(i % 10), ha="right")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("value")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{label}: operating metrics vs threshold", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def ece_before_after(
    ece_raw: Dict[str, float], ece_cal: Dict[str, float], path: Path,
    split: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    labels = list(ece_raw)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, [ece_raw[l] for l in labels], 0.36, label="raw", color="#c0392b")
    ax.bar(x + 0.18, [ece_cal[l] for l in labels], 0.36, label="temperature-scaled",
           color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ECE (15 equal-width bins)")
    ax.set_title(f"Expected calibration error — {split}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Error analysis (Phase 7)
# --------------------------------------------------------------------------- #
def strata_bars(rows: Sequence[Dict[str, object]], path: Path) -> Path:
    """TP/FP/FN/TN counts per policy (mean over labels) — error anatomy."""
    import collections

    policies = list(dict.fromkeys(r["policy"] for r in rows))
    strata = ("TP", "FP", "FN", "TN")
    agg: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows:
        for s in strata:
            agg[r["policy"]][s] += float(r[f"n_{s.lower()}"]) / max(len(rows) / len(policies), 1)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(policies))
    width = 0.2
    colors = {"TP": "#27ae60", "FP": "#c0392b", "FN": "#e67e22", "TN": "#95a5a6"}
    for i, s in enumerate(strata):
        ax.bar(x + (i - 1.5) * width, [agg[p][s] for p in policies], width,
               label=s, color=colors[s])
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=15, ha="right")
    ax.set_ylabel("images (mean over labels)")
    ax.set_title("Confusion strata by frozen policy", fontsize=10)
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def error_rate_by_confidence(
    rows: Sequence[Dict[str, object]], title: str, path: Path
) -> Path:
    """Error rate and sample mass per confidence bin (10 equal-width bins)."""
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    have = [r for r in rows if r["count"]]
    centers = [0.5 * (r["lo"] + r["hi"]) for r in have]
    width = (have[0]["hi"] - have[0]["lo"]) * 0.9 if have else 0.1
    ax1.bar(centers, [r["count"] for r in have], width=width, alpha=0.25,
            color="#34495e", label="images")
    ax1.set_xlabel("predicted probability (bin)")
    ax1.set_ylabel("images", color="#34495e")
    ax2 = ax1.twinx()
    ax2.plot(centers, [r["error_rate"] for r in have], marker="o", linewidth=1.8,
             color="#c0392b", label="error rate")
    ax2.set_ylabel("error rate", color="#c0392b")
    ax2.set_ylim(0, 1)
    ax1.set_title(title, fontsize=10)
    lines = ax1.get_lines() or []
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    ax1.grid(alpha=0.3)
    return _save(fig, path)
