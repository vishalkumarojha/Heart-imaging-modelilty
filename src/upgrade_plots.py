"""IEEE-upgrade figures (papillary set 10..14).

Adds the five statistical deliverables to the deterministic figure library:

    fig_9_calibration_comparison.png   → paper figure_10  reliability diagram
                                         (raw vs temperature vs logistic)
    fig_10_ece_sensitivity.png         → paper figure_11  ECE across binning grid
    fig_11_threshold_stability.png     → paper figure_12  threshold CIs (val bootstrap)
    fig_12_decision_policy.png         → paper figure_13  test F1 policy × calibration
    fig_13_prevalence_sensitivity.png  → paper figure_14  prevalence-shift simulation

All figures are derived ONLY from the frozen artifacts (statistics loaders,
ece_sensitivity.json, threshold_stability.json, decision_policy_analysis.csv and
prevalence_shift.json).  Nothing is re-measured here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402
from .calibration import reliability_bins  # noqa: E402
from .metrics import binary_metrics  # noqa: E402
from .statistics.data import load_test_predictions, variant_probs  # noqa: E402

DPI = 150
NAVY = "#1f4e79"
RED = "#7a0f1f"
ORANGE = "#d9822b"
PURPLE = "#8e44ad"
GREEN = "#2e7d32"
GREY = "#7f8c8d"

VARIANTS = ("raw", "calibrated", "logistic")
_VARIANT_STYLE = {
    "raw": dict(color=RED, label="raw logit"),
    "calibrated": dict(color=ORANGE, label="temperature (T=1.19 / 1.40)"),
    "logistic": dict(color=GREEN, label="Platt logistic (slope>0)"),
}

SKIPPED_POLICIES = ("fixed",)  # trivially static — excluded from stability figure


def _save(fig, name: str) -> Path:
    path = C.FINAL_FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_json(rel: Path) -> dict:
    if not rel.exists():
        raise FileNotFoundError(f"required artifact missing: {rel}")
    return json.loads(rel.read_text())


# --------------------------------------------------------------------------- #
# fig_9  →  figure_10  calibration comparison (reliability diagram, per label)
# --------------------------------------------------------------------------- #
def figure_calibration_comparison() -> Path:
    df = load_test_predictions()
    fig, axes = plt.subplots(1, len(C.TARGET_LABELS), figsize=(10.6, 4.6))
    fig.patch.set_facecolor("white")
    if len(C.TARGET_LABELS) == 1:
        axes = [axes]
    for ax, label in zip(axes, C.TARGET_LABELS):
        y = df[f"true_{label}"].to_numpy(dtype=np.float64)
        ax.plot([0, 1], [0, 1], linestyle="--", color=GREY, linewidth=1,
                label="perfectly calibrated")
        for variant in VARIANTS:
            p = variant_probs(df, label, variant)
            b = reliability_bins(y, p, n_bins=C.CALIBRATION_BINS)
            ok = b["count"] > 0
            style = _VARIANT_STYLE[variant]
            ax.plot(b["mean_confidence"][ok], b["empirical_accuracy"][ok],
                    marker="o", markersize=4, linewidth=1.5,
                    color=style["color"], label=style["label"])
        ax.set_xlabel("mean predicted probability (bin)")
        ax.set_ylabel("observed frequency (bin)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label} (test)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Calibration comparison on the frozen test split "
                 "(15 equal-width bins)", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig_9_calibration_comparison.png")


# --------------------------------------------------------------------------- #
# fig_10  →  figure_11  ECE sensitivity across the binning grid
# --------------------------------------------------------------------------- #
def figure_ece_sensitivity() -> Path:
    data = _load_json(C.ECE_SENSITIVITY_JSON)
    fig, axes = plt.subplots(1, len(C.TARGET_LABELS), figsize=(10.0, 3.9))
    fig.patch.set_facecolor("white")
    if len(C.TARGET_LABELS) == 1:
        axes = [axes]
    for ax, label in zip(axes, C.TARGET_LABELS):
        cells = [r for r in data["rows"] if r["class"] == label]
        x = np.arange(len(VARIANTS))
        means, los, his = [], [], []
        for j, variant in enumerate(VARIANTS):
            vals = [r["ece"] for r in cells if r["calibration"] == variant]
            means.append(float(np.mean(vals)))
            los.append(float(np.min(vals)))
            his.append(float(np.max(vals)))
        ref = [data["ece_at_reference_binning"][label].get(v) for v in VARIANTS]
        for i, (m, lo_, hi_, r) in enumerate(zip(means, los, his, ref)):
            ax.errorbar(x[i], m, yerr=[[max(0.0, m - lo_)], [max(0.0, hi_ - m)]],
                        fmt="o",
                        color=_VARIANT_STYLE[VARIANTS[i]]["color"],
                        markersize=6, capsize=4, linewidth=1.4)
            ax.plot(x[i], r, marker="*", color="black", markersize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([v.capitalize() for v in VARIANTS], fontsize=8)
        ax.set_ylabel("ECE (lower is better)")
        ax.set_ylim(0, max(his) * 1.25)
        ax.set_title(f"{label}: ECE across bins {tuple(data['bins'])} × "
                     f"strategies {tuple(data['strategies'])}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.995, 0.02,
                "dot: mean over grid cells · error bar: min–max · star: reference "
                "binning (15, equal-width)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, color=GREY)
    fig.tight_layout()
    return _save(fig, "fig_10_ece_sensitivity.png")


# --------------------------------------------------------------------------- #
# fig_11  →  figure_12  threshold stability (val patient-level bootstrap)
# --------------------------------------------------------------------------- #
def figure_threshold_stability() -> Path:
    data = _load_json(C.THRESHOLD_STABILITY_JSON)
    policies = [p for p in C.THRESHOLD_POLICIES if p not in SKIPPED_POLICIES]
    width = 0.26
    fig, axes = plt.subplots(1, len(C.TARGET_LABELS), figsize=(11.4, 4.0))
    fig.patch.set_facecolor("white")
    if len(C.TARGET_LABELS) == 1:
        axes = [axes]
    for ax, label in zip(axes, C.TARGET_LABELS):
        per_label = data["per_label"][label]
        x0 = np.arange(len(policies))
        for j, variant in enumerate(VARIANTS):
            per_variant = per_label[variant]
            first_fit, los, his = [], [], []
            for pol in policies:
                rec = per_variant[pol]
                first_fit.append(rec["first_fit"])
                los.append(rec["ci_lower"])
                his.append(rec["ci_upper"])
            ax.bar(x0 + (j - 1.0) * width, first_fit, width * 0.9,
                   color=_VARIANT_STYLE[variant]["color"],
                   label=_VARIANT_STYLE[variant]["label"], alpha=0.9)
            ax.bar(x0 + (j - 1.0) * width, [h - l for l, h in zip(los, his)],
                   width * 0.9, bottom=los, color="none", edgecolor="black",
                   linewidth=0.8, alpha=0.35)
        ax.axhline(0.5, linestyle="--", color=GREY, linewidth=1,
                   label="fixed τ=0.50")
        ax.set_xticks(x0)
        ax.set_xticklabels([p.replace("_", "\n") for p in policies], fontsize=7)
        ax.set_ylabel("threshold τ")
        ax.set_title(f"{label}: val patient-bootstrap τ (n={data['n_bootstraps']})",
                     fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Threshold stability: refit per validation patient-bootstrap "
                 "(bar: first fit · interval: percentile CI)", fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig_11_threshold_stability.png")


# --------------------------------------------------------------------------- #
# fig_12  →  figure_13  decision-policy matrix (test F1)
# --------------------------------------------------------------------------- #
def figure_decision_policy() -> Path:
    df = pd.read_csv(C.DECISION_POLICY_ANALYSIS_CSV)
    policies = list(C.THRESHOLD_POLICIES)
    fig, axes = plt.subplots(1, len(C.TARGET_LABELS), figsize=(10.6, 4.2))
    fig.patch.set_facecolor("white")
    if len(C.TARGET_LABELS) == 1:
        axes = [axes]
    cmap = plt.get_cmap("YlGnBu")
    for ax, label in zip(axes, C.TARGET_LABELS):
        sub = df[df["class"] == label]
        mat = np.zeros((len(VARIANTS), len(policies)))
        for i, v in enumerate(VARIANTS):
            for j, p in enumerate(policies):
                row = sub[(sub["variant"] == v) & (sub["policy"] == p)]
                mat[i, j] = float(row["test_f1"].iloc[0])
        vmin, vmax = mat.min(), mat.max()
        ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(policies)))
        ax.set_xticklabels([p.replace("_", "\n") for p in policies], fontsize=7)
        ax.set_yticks(np.arange(len(VARIANTS)))
        ax.set_yticklabels([v.capitalize() for v in VARIANTS], fontsize=8)
        for i in range(len(VARIANTS)):
            for j in range(len(policies)):
                ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if vmax and mat[i, j] > 0.5 * (vmin + vmax)
                        else "black")
        ax.set_title(f"{label} (test)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("threshold policy")
    fig.suptitle("Test F1 by calibration space × frozen decision policy",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig_12_decision_policy.png")


# --------------------------------------------------------------------------- #
# fig_13  →  figure_14  prevalence-shift sensitivity
# --------------------------------------------------------------------------- #
def figure_prevalence_sensitivity() -> Path:
    data = _load_json(C.PREVALENCE_SHIFT_JSON)
    df = load_test_predictions()
    fig, axes = plt.subplots(1, len(C.TARGET_LABELS), figsize=(11.0, 4.2))
    fig.patch.set_facecolor("white")
    if len(C.TARGET_LABELS) == 1:
        axes = [axes]
    for ax, label in zip(axes, C.TARGET_LABELS):
        obs = data["per_label"][label]["observed_prevalence"]
        tau = float(data["per_label"][label]["threshold_calibrated_f1_optimal"])
        y = df[f"true_{label}"].to_numpy(dtype=np.float64)
        m_obs = binary_metrics(y, variant_probs(df, label, "calibrated"), tau)

        cohorts = [c for c in data["per_label"][label]["cohorts"]
                   if c.get("simulated")]
        prevs = [c["observed_prevalence"] * 100 for c in cohorts]
        f1 = [c["f1"] for c in cohorts]
        prec = [c["precision"] for c in cohorts]
        auprc = [c["auprc"] for c in cohorts]

        for xv, yv, color, name in ((prevs, f1, NAVY, "F1"),
                                    (prevs, prec, ORANGE, "precision"),
                                    (prevs, auprc, PURPLE, "AUPRC")):
            ax.plot(xv, yv, marker="o", linewidth=1.5, color=color, label=name)
        ax.plot([obs * 100], [m_obs["f1"]], marker="D", color=RED,
                markersize=7, label=f"observed {obs:.3f} prevalence")
        ax.axvline(obs * 100, linestyle=":", color=GREY, linewidth=1)
        ax.set_xlabel("simulated test prevalence (%)")
        ax.set_ylabel("metric value")
        ax.set_title(f"{label}: frozen arm-D threshold across prevalence",
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Prevalence-shift simulation (negative subsampling of frozen "
                 "test predictions; recall is constant by construction)",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, "fig_13_prevalence_sensitivity.png")


def build_ieee_figures() -> List[Path]:
    written = [
        figure_calibration_comparison(),
        figure_ece_sensitivity(),
        figure_threshold_stability(),
        figure_decision_policy(),
        figure_prevalence_sensitivity(),
    ]
    for p in written:
        if not p.exists():
            raise RuntimeError(f"figure was not written: {p}")
    return written


if __name__ == "__main__":
    for path in build_ieee_figures():
        print(path)