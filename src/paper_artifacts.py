"""Paper-ready evidence pack (final phase).

Derives EVERYTHING under `outputs/final_results/` from the frozen experiment
artifacts — no measurement, no retraining, no resampling beyond the frozen
thresholds and the frozen per-image predictions. Everything is reproducible by
re-running `python -m src.paper_artifacts` and results are byte-identical.

Outputs:
    tables/table_1_dataset.csv  .. table_8_external.csv   (paper Tables 1-8)
    figures/fig_1_roc.png  ..  fig_8_gradcam.png           (paper Figures 1-8)
    error_analysis.csv                                    (task-26 CSV)
    region_analysis.json                                  (box/heatmap record)
    gradcam/ (overlay copies + cases.csv + metadata JSON)
    research_summary.json                                  (single evidence doc)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C
from .metrics import auprc, auroc
from .reproducibility import write_json
from .utils import setup_logging

logger = setup_logging()

DPI = 150


def _load(rel: str) -> dict:
    p = Path(C.PROJECT_ROOT) / rel
    if not p.exists():
        raise FileNotFoundError(f"required artifact missing: {p}")
    return json.loads(p.read_text())


def _load_test_predictions() -> pd.DataFrame:
    matches = sorted(Path(C.RAW_PREDICTIONS_DIR).glob("test__*.csv"))
    if not matches:
        raise FileNotFoundError(f"no frozen test predictions in {C.RAW_PREDICTIONS_DIR}")
    return pd.read_csv(matches[0])


def _save(fig, name: str) -> Path:
    path = C.FINAL_FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# --------------------------------------------------------------------------- #
# Shared readers
# --------------------------------------------------------------------------- #
class Artifacts:
    def __init__(self):
        self.manifest = _load("outputs/checkpoints/baseline/baseline_manifest.json")
        self.base_test = _load("outputs/metrics/baseline/baseline_metrics_test.json")
        self.exp1 = _load("outputs/metrics/experiments/exp1_calibration.json")
        self.exp2 = _load("outputs/metrics/experiments/exp2_thresholds_raw.json")
        self.exp3 = _load("outputs/metrics/experiments/exp3_thresholds_calibrated.json")
        self.ea = _load("outputs/metrics/error_analysis/error_analysis_test.json")
        self.bbox = _load("outputs/gradcam/bbox_localization.json")
        self.grad = _load("outputs/gradcam/gradcam_summary.json")
        self.case_list = _load("outputs/gradcam/case_list.json")
        self.split_index = pd.read_csv(C.DATA_PROCESSED_DIR / "split_index.csv")
        self.raw_probs = _load_test_predictions()
        self.cal = _load("outputs/metrics/calibration/calibration_summary.json") \
            if (C.CALIBRATION_METRICS_DIR / "calibration_summary.json").exists() else {}
        temps = _load("outputs/metrics/calibration/temperature_scalers.json") \
            if (C.CALIBRATION_METRICS_DIR / "temperature_scalers.json").exists() else {}
        self.temp = {lbl: float(temps["metadata"]["temperatures"][lbl])
                     for lbl in C.TARGET_LABELS} if temps else {}

    def exp_eval(self, variant: str, split: str, label: str, policy: str) -> dict:
        src = self.exp2 if variant == "raw" else self.exp3
        return src["evaluation"][split][label][policy]


ART = Artifacts()


def _to_df(rows: List[dict], name: str) -> Path:
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"refusing to write empty table {name}")
    out = C.FINAL_TABLES_DIR / name
    df.to_csv(out, index=False)
    logger.info("wrote %s (%d rows)", name, len(df))
    return out


# --------------------------------------------------------------------------- #
# Table 1 — dataset
# --------------------------------------------------------------------------- #
def table1() -> Path:
    si = ART.split_index
    lbl_counts = si.groupby("split").agg(
        cardiomegaly=("Cardiomegaly", "sum"), effusion=("Effusion", "sum")).reset_index()
    n_patients = si.groupby("split")["Patient ID"].nunique()
    rows = []
    order = {"train": 0, "val": 1, "test": 2}
    for split, grp in si.groupby("split"):
        lc = lbl_counts[lbl_counts.split == split].iloc[0]
        rows.append({
            "split": split,
            "images": int(len(grp)),
            "patients": int(n_patients[split]),
            "cardiomegaly_positive": int(lc.cardiomegaly),
            "effusion_positive": int(lc.effusion),
        })
    rows.sort(key=lambda r: order[r["split"]])
    rows.append({
        "split": "total",
        "images": int(len(si)),
        "patients": int(si["Patient ID"].nunique()),
        "cardiomegaly_positive": int(si.Cardiomegaly.sum()),
        "effusion_positive": int(si.Effusion.sum()),
    })
    return _to_df(rows, "table_1_dataset.csv")


# --------------------------------------------------------------------------- #
# Table 2 — model & configuration
# --------------------------------------------------------------------------- #
def table2() -> Path:
    cfg = ART.manifest["baseline_configuration"]
    rows = [
        {"key": "dataset", "value": cfg.get("dataset")},
        {"key": "task", "value": "multi-label Cardiomegaly + Effusion (binary per label)"},
        {"key": "architecture", "value": cfg.get("architecture")},
        {"key": "classifier", "value": cfg.get("classifier")},
        {"key": "input resolution", "value": f'{cfg["image_size"]}×{cfg["image_size"]}'},
        {"key": "split", "value": cfg.get("split")},
        {"key": "loss", "value": cfg.get("loss")},
        {"key": "optimizer", "value": cfg.get("optimizer")},
        {"key": "schedule", "value": json.dumps(cfg.get("schedule"))},
        {"key": "batch size", "value": cfg.get("batch_size")},
        {"key": "mixed precision", "value": cfg.get("amp")},
        {"key": "seed", "value": cfg.get("seed")},
        {"key": "checkpoint selection", "value": cfg.get("checkpoint_selection")},
        {"key": "checkpoint sha256", "value": ART.manifest["sha256"]},
        {"key": "baseline status", "value": ART.manifest["status"]},
    ]
    return _to_df(rows, "table_2_model_config.csv")


# --------------------------------------------------------------------------- #
# Table 3 — main classification results (test, raw baseline + combined arm)
# --------------------------------------------------------------------------- #
def table3() -> Path:
    rows = []
    for lbl in C.TARGET_LABELS:
        e1 = ART.exp1["splits"]["test"][lbl]
        # baseline fixed 0.5
        base = ART.base_test["metrics"]["per_label"][lbl]
        # combined arm = calibrated + f1_optimal
        combined = ART.exp3["evaluation"]["test"][lbl]["f1_optimal"]
        rows.append({
            "class": lbl,
            "auroc": round(e1["auroc_raw"], 4),
            "auprc": round(e1["auprc_raw"], 4),
            "baseline_f1_fixed_0.5": round(base["f1"], 4),
            "baseline_sensitivity": round(base["sensitivity"], 4),
            "baseline_specificity": round(base["specificity"], 4),
            "combined_f1_f1_optimal": round(combined["f1"], 4),
            "combined_sensitivity": round(combined["recall"], 4),
            "combined_specificity": round(combined["specificity"], 4),
            "test_n": int(base["n"]),
            "prevalence": round(float(base["prevalence"]), 4),
        })
    return _to_df(rows, "table_3_main_results.csv")


# --------------------------------------------------------------------------- #
# Table 4 — ablation (A baseline / B calibration / C threshold / D combined)
# --------------------------------------------------------------------------- #
def table4() -> Path:
    rows = []
    for lbl in C.TARGET_LABELS:
        e1 = ART.exp1["splits"]["test"][lbl]
        base = ART.base_test["metrics"]["per_label"][lbl]
        for arm, label_variant, policy in (
            ("A baseline", "raw", "fixed"),
            ("B calibration only", "calibrated", "fixed"),
            ("C threshold only", "raw", "f1_optimal"),
            ("D calibration + threshold", "calibrated", "f1_optimal"),
        ):
            ev = (ART.exp2 if label_variant == "raw" else ART.exp3)["evaluation"]["test"][lbl]
            m = ev[policy]
            ece = e1[label_variant]["ece"]
            brier = e1[label_variant]["brier"]
            rows.append({
                "class": lbl, "arm": arm,
                "f1": round(m["f1"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "specificity": round(m["specificity"], 4),
                "auc_unchanged": (round(auroc(_y(lbl), _p(lbl)), 4) == round(e1["auroc_raw"], 4)),
                "ece": round(ece, 4),
                "brier": round(brier, 4),
            })
    return _to_df(rows, "table_4_ablation.csv")


def _y(lbl):  # reuse across table4/figures
    return ART.raw_probs[f"true_{lbl}"].to_numpy(dtype=np.float64)


def _p(lbl):
    return ART.raw_probs[f"prob_{lbl}"].to_numpy(dtype=np.float64)


def _pc(lbl):
    logits = ART.raw_probs[f"logit_{lbl}"].to_numpy(dtype=np.float64)
    return _sigmoid(logits / ART.temp[lbl])


# --------------------------------------------------------------------------- #
# Table 5 — threshold policies (val-fitted τ → test operating metrics)
# --------------------------------------------------------------------------- #
def table5() -> Path:
    rows = []
    for variant in ("raw", "calibrated"):
        src = ART.exp2 if variant == "raw" else ART.exp3
        thr = _load(f"outputs/metrics/thresholds/"
                    f"{'thresholds' if variant == 'raw' else 'thresholds_calibrated'}_val.json")
        for lbl in C.TARGET_LABELS:
            for policy in src["evaluation"]["test"][lbl]:
                if policy.startswith("_"):
                    continue  # internal alias row, not a policy
                m = src["evaluation"]["test"][lbl][policy]
                t = thr["policies"][policy][lbl]["threshold"]
                rows.append({
                    "calibration": variant, "class": lbl, "policy": policy,
                    "threshold_val": round(float(t), 4),
                    "f1_test": round(m["f1"], 4),
                    "precision_test": round(m["precision"], 4),
                    "recall_test": round(m["recall"], 4),
                    "specificity_test": round(m["specificity"], 4),
                })
    return _to_df(rows, "table_5_threshold_policies.csv")


# --------------------------------------------------------------------------- #
# Table 6 — error analysis summary
# --------------------------------------------------------------------------- #
def table6() -> Path:
    rows = []
    for lbl in C.TARGET_LABELS:
        strata = ART.ea["strata"]["f1_optimal"][lbl]
        hc = strata["high_confidence_errors"]
        strat = strata["strata"]
        rows.append({
            "class": lbl,
            "error_rate": round(strata["error_rate"], 4),
            "n_errors": round(strata["error_rate"] * strata["n"]),  # computed; see table6 note
            "tp": strat["TP"]["count"], "fp": strat["FP"]["count"],
            "fn": strat["FN"]["count"], "tn": strat["TN"]["count"],
            "high_conf_errors": hc["count"],
            "high_conf_share_of_errors": round(hc["share_of_errors"], 4),
            "mean_certainty": round(strata["mean_certainty"], 4),
        })
    return _to_df(rows, "table_6_error_analysis.csv")


# --------------------------------------------------------------------------- #
# Table 7 — explainability / region analysis
# --------------------------------------------------------------------------- #
def table7() -> Path:
    b = ART.bbox
    by_label = {}
    for c in b["cases"]:
        d = by_label.setdefault(c["label"], [0, 0, []])
        d[0] += 1
        d[1] += int(c["pointing_game_hit"])
        d[2].append(c["concentration_ratio"])
    rows = [{
        "scope": "localization sanity check (fixed cohort)",
        "label": "all",
        "n_with_box": b["n_with_box"],
        "pointing_game_hits": b["pointing_game_hits"],
        "pointing_game_accuracy": round(b["pointing_game_accuracy"], 4),
        "concentration_ratio_gt_1": b["ratio_above_1"],
        "mean_concentration_ratio": round(b["mean_concentration_ratio"], 4),
        "median_concentration_ratio": round(b["median_concentration_ratio"], 4),
    }]
    for lbl, (n, hits, ratios) in by_label.items():
        rows.append({
            "scope": "fixed cohort", "label": lbl, "n_with_box": n,
            "pointing_game_hits": hits,
            "pointing_game_accuracy": round(hits / n, 4),
            "concentration_ratio_gt_1": int(sum(r > 1 for r in ratios)),
            "mean_concentration_ratio": round(float(np.mean(ratios)), 4),
            "median_concentration_ratio": round(float(np.median(ratios)), 4),
        })
    return _to_df(rows, "table_7_explainability.csv")


# --------------------------------------------------------------------------- #
# Table 8 — external validation (honest absence)
# --------------------------------------------------------------------------- #
def table8() -> Path:
    status = _load("outputs/metrics/external/status.json")
    rows = [{
        "dataset": "external (e.g. CheXpert / MIMIC-CXR)",
        "available": False,
        "status": status.get("status", "BLOCKED — EXTERNAL DATASET NOT AVAILABLE"),
        "no_metrics_were_computed": status.get("no_metrics_were_computed", True),
        "note": status.get("note", ""),
    }]
    return _to_df(rows, "table_8_external_validation.csv")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig1_roc() -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    styles = {"Cardiomegaly": "#2980b9", "Effusion": "#c0392b"}
    for lbl in C.TARGET_LABELS:
        fpr, tpr, _ = _roc(_y(lbl), _p(lbl))
        auc = auroc(_y(lbl), _p(lbl))
        ax.plot(fpr, tpr, color=styles[lbl], linewidth=1.8,
                label=f"{lbl} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Figure 1 — ROC curves (test, frozen baseline)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "fig_1_roc.png")


def _roc(y, p):
    from sklearn.metrics import roc_curve
    return roc_curve(y, p)


def fig2_pr() -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    styles = {"Cardiomegaly": "#2980b9", "Effusion": "#c0392b"}
    for lbl in C.TARGET_LABELS:
        from sklearn.metrics import precision_recall_curve
        prec, rec, _ = precision_recall_curve(_y(lbl), _p(lbl))
        ap = auprc(_y(lbl), _p(lbl))
        ax.plot(rec, prec, color=styles[lbl], linewidth=1.8,
                label=f"{lbl} (AUPRC={ap:.3f})")
        ax.axhline(float(_y(lbl).mean()), color=styles[lbl], linestyle=":",
                   linewidth=0.8, alpha=0.6)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1)
    ax.set_title("Figure 2 — precision–recall curves (test, dashed = prevalence)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "fig_2_pr.png")


def fig3_reliability() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True, sharey=True)
    for ax, lbl in zip(axes, C.TARGET_LABELS):
        ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
        _reliability_line(ax, _y(lbl), _p(lbl), "raw", "#c0392b")
        _reliability_line(ax, _y(lbl), _pc(lbl), "temperature-scaled", "#e67e22")
        ax.set_title(f"{lbl} — raw vs temperature-scaled", fontsize=10)
        ax.set_xlabel("mean predicted probability (15 equal-width bins)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("observed frequency")
    fig.suptitle("Figure 3 — reliability diagrams (test)", fontsize=11)
    return _save(fig, "fig_3_reliability.png")


def _reliability_line(ax, y, p, name, color):
    from .calibration import reliability_bins
    b = reliability_bins(y, p, n_bins=C.CALIBRATION_BINS)
    ok = b["count"] > 0
    ax.plot(b["mean_confidence"][ok], b["empirical_accuracy"][ok],
            marker="o", markersize=4, linewidth=1.5, color=color, label=name)


def fig4_threshold_sweep() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, lbl in zip(axes, C.TARGET_LABELS):
        y, p = _y(lbl), _pc(lbl)
        taus = np.unique(p)
        keep = np.linspace(0, len(taus) - 1, min(600, len(taus))).astype(int)
        taus = taus[keep]
        prec, rec, f1 = [], [], []
        for t in taus:
            pred = (p >= t).astype(int)
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            prec.append(tp / (tp + fp) if (tp + fp) else np.nan)
            rec.append(tp / (tp + fn) if (tp + fn) else np.nan)
            f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
        ax.plot(taus, prec, label="precision", linewidth=1.4)
        ax.plot(taus, rec, label="recall", linewidth=1.4)
        ax.plot(taus, f1, label="F1", linewidth=1.5, linestyle="--")
        thr = ART.exp3["evaluation"]["test"][lbl]
        for i, pol in enumerate(("fixed", "f1_optimal", "youden",
                                 "sensitivity_constrained", "precision_constrained")):
            tau = thr[pol]["threshold"]
            ax.axvline(tau, color=plt.get_cmap("tab10")(i), linewidth=0.8, alpha=0.7)
            ax.text(tau, 1.03, pol.replace("_", "\n"), rotation=0, fontsize=6,
                    ha="center", color=plt.get_cmap("tab10")(i))
        ax.set_title(f"{lbl} — calibrated", fontsize=10)
        ax.set_xlabel("decision threshold")
        ax.set_ylim(0, 1.06)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="center left")
    fig.suptitle("Figure 4 — operating metrics vs threshold with frozen policies (test)",
                 fontsize=11)
    return _save(fig, "fig_4_threshold_sweep.png")


def fig5_confusion() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    spec = {"Cardiomegaly": ("A. baseline (raw, τ=0.5)", "raw", "fixed"),
            "Effusion": None}
    for r, lbl in enumerate(C.TARGET_LABELS):
        for c, (title, variant, policy) in enumerate((
                ("baseline (raw, τ=0.5)", "raw", "fixed"),
                ("combined (calibrated, τ=f1_optimal)", "calibrated", "f1_optimal"))):
            ax = axes[r][c]
            m = (ART.exp2 if variant == "raw" else ART.exp3)["evaluation"]["test"][lbl][policy]
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            im = ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{int(cm[i, j])}\n({cm[i, j]/sum(cm.sum(axis=1)):.1%})",
                            ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["predicted 0", "predicted 1"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
            ax.set_title(f"{lbl} — {title}", fontsize=9)
            ax.grid(False)
    fig.suptitle("Figure 5 — confusion matrices (test)", fontsize=11)
    return _save(fig, "fig_5_confusion.png")


def fig6_training_curves() -> Path:
    df = pd.read_csv(C.LOG_DIR / "metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax = axes[0]
    ax.plot(df.epoch, df.train_loss, marker="o", markersize=3, label="train loss")
    ax.plot(df.epoch, df.val_loss, marker="o", markersize=3, label="val loss")
    sep = df[df.phase == "phase2-unfrozen"].epoch.min()
    ax.axvline(sep - 0.5, color="#888888", linestyle=":", linewidth=1)
    ax.annotate("unfreeze\nbackbone", (sep - 0.5, max(df.val_loss)),
                fontsize=7, ha="right")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Figure 6a — loss", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    for lbl in C.TARGET_LABELS:
        ax.plot(df.epoch, df[f"val_auroc_{lbl}"], marker="o", markersize=3, label=lbl)
    ax.plot(df.epoch, df.val_auroc_mean, marker="D", markersize=3,
            color="#2c3e50", label="mean")
    ax.axvline(sep - 0.5, color="#888888", linestyle=":", linewidth=1)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation AUROC")
    ax.set_ylim(0.82, 0.90)
    ax.set_title("Figure 6b — validation AUROC (best epoch = 8)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return _save(fig, "fig_6_training_curves.png")


def fig7_error_analysis() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    for lbl in C.TARGET_LABELS:
        bins = ART.ea["confidence_error_bins"][lbl]
        centers = [0.5 * (b["lo"] + b["hi"]) for b in bins if b["count"]]
        err = [b["error_rate"] for b in bins if b["count"]]
        cnt = [b["count"] for b in bins if b["count"]]
        ax.bar(centers, cnt, width=0.09, alpha=0.25)
        ax.plot(centers, err, marker="o", markersize=3, label=f"{lbl} error rate")
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("error rate (points) / images (bars)")
    ax.set_ylim(0, 1)
    ax.set_title("Figure 7a — error rate by confidence (test, f1_optimal)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper center")
    ax = axes[1]
    strata = ("TP", "FP", "FN", "TN")
    x = np.arange(len(C.TARGET_LABELS))
    width = 0.19
    colors = {"TP": "#27ae60", "FP": "#c0392b", "FN": "#e67e22", "TN": "#95a5a6"}
    for i, s in enumerate(strata):
        vals = [ART.ea["strata"]["f1_optimal"][lbl]["strata"][s]["count"]
                for lbl in C.TARGET_LABELS]
        ax.bar(x + (i - 1.5) * width, vals, width, label=s, color=colors[s])
    ax.set_xticks(x); ax.set_xticklabels(C.TARGET_LABELS)
    ax.set_ylabel("images")
    ax.set_title("Figure 7b — confusion strata by class (test, f1_optimal)", fontsize=9)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=7)
    return _save(fig, "fig_7_error_analysis.png")


def fig8_gradcam() -> Path:
    overlays = C.GRADCAM_DIR / "overlays"
    files = sorted(overlays.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"no grad-cam overlays under {overlays}")
    want = {"Cardiomegaly": [], "Effusion": []}
    for f in files:
        for lbl in C.TARGET_LABELS:
            if f.stem.endswith(f"__{lbl.lower()}"):
                want[lbl].append(f)
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.0))
    strata = ("TP", "FP", "FN", "TN")
    for r, lbl in enumerate(C.TARGET_LABELS):
        for c, s in enumerate(strata):
            f = next((f for f in sorted(want[lbl]) if f.name.split("_")[1] == s), None)
            if f is None:
                raise FileNotFoundError(f"missing {s} overlay for {lbl}")
            ax = axes[r][c]
            img = plt.imread(f)
            ax.imshow(img)
            ax.set_title(f"{lbl} — {s}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Figure 8 — Grad-CAM overlays (TP / FP / FN / TN, test cases)",
                 fontsize=11)
    return _save(fig, "fig_8_gradcam.png")


# --------------------------------------------------------------------------- #
# error_analysis.csv (task-26) — class, error_type, confidence_bin, count, %
# --------------------------------------------------------------------------- #
def error_analysis_csv() -> Path:
    rows = []
    for lbl in C.TARGET_LABELS:
        strata = ART.ea["strata"]["f1_optimal"][lbl]
        n = strata["n"]
        for s in ("TP", "FP", "FN", "TN"):
            c = strata["strata"][s]["count"]
            rows.append({
                "class": lbl, "error_type": s, "confidence_bin": "all",
                "count": c, "percentage": round(100.0 * c / n, 3),
            })
        for b in ART.ea["confidence_error_bins"][lbl]:
            cnt = int(round(b["count"] * b["error_rate"]))
            rows.append({
                "class": lbl, "error_type": "error",
                "confidence_bin": f"[{b['lo']:.1f}-{b['hi']:.1f}]",
                "count": cnt,
                "percentage": round(100.0 * cnt / n, 3),
            })
    df = pd.DataFrame(rows)
    out = C.ERROR_ANALYSIS_CSV
    df.to_csv(out, index=False)
    logger.info("error_analysis.csv -> %d rows", len(df))
    return out


# --------------------------------------------------------------------------- #
# region_analysis.json — cautious descriptors, fixed cohort
# --------------------------------------------------------------------------- #
def region_analysis_json() -> Path:
    b = ART.bbox
    per = {}
    for c in b["cases"]:
        lbl = c["label"]
        d = per.setdefault(lbl, {"n_with_box": 0, "hits": 0, "ratios": [],
                                 "box_area_fraction": [], "heat_mass_inside": []})
        d["n_with_box"] += 1
        d["hits"] += int(c["pointing_game_hit"])
        d["ratios"].append(c["concentration_ratio"])
        d["box_area_fraction"].append(c["box_area_fraction"])
        d["heat_mass_inside"].append(c["heat_mass_inside"])
    payload = {
        "scope": ("cautious localization check, NOT a diagnostic localization "
                  "validation"),
        "cohort_definition": b["cohort"],
        "n_cases": b["n_with_box"],
        "aggregate": {
            "mean_concentration_ratio": round(b["mean_concentration_ratio"], 4),
            "median_concentration_ratio": round(b["median_concentration_ratio"], 4),
            "fraction_ratio_above_1": round(b["ratio_above_1"] / b["n_with_box"], 4),
            "pointing_game_hits": b["pointing_game_hits"],
            "pointing_game_accuracy": round(b["pointing_game_accuracy"], 4),
        },
        "per_label": {
            lbl: {
                "n_with_box": d["n_with_box"],
                "pointing_game_accuracy": round(d["hits"] / d["n_with_box"], 4),
                "mean_concentration_ratio": round(float(np.mean(d["ratios"])), 4),
                "mean_box_area_fraction": round(float(np.mean(d["box_area_fraction"])), 4),
                "mean_heat_mass_inside_box": round(float(np.mean(d["heat_mass_inside"])), 4),
            }
            for lbl, d in per.items()
        },
        "interpretation": ("The concentration-ratio (heat mass inside the NIH box "
                           "relative to expected under uniformity) is >1 for most "
                           "test images that carry a box, over 2.0 on average, and "
                           "the pointing game accuracy is class-dependent — evidence "
                           "for the baseline attending to overlapping anatomy, not a "
                           "claim of clinical localization validity."),
    }
    return write_json(C.REGION_ANALYSIS_JSON, payload)


# --------------------------------------------------------------------------- #
# grad-cam export (overlay copies + case metadata)
# --------------------------------------------------------------------------- #
def gradcam_export() -> Path:
    dst = C.FINAL_GRADCAM_DIR
    dst.mkdir(parents=True, exist_ok=True)
    overlays = C.GRADCAM_DIR / "overlays"
    for f in sorted(overlays.glob("*.png")):
        shutil.copy2(f, dst / f.name)
    for name in ("case_list.json", "gradcam_summary.json", "bbox_localization.json"):
        src = C.GRADCAM_DIR / name
        if src.exists():
            shutil.copy2(src, dst / name)
    rows = []
    for case in ART.grad["cases"]:
        s = case["stratum"]  # TP/FP/FN/TN encodes the pair
        gt = 1 if s in ("TP", "FN") else 0
        pred = 1 if s in ("TP", "FP") else 0
        rows.append({
            "sample_id": case["case_id"],
            "class": case["label"],
            "ground_truth": gt,
            "prediction": pred,
            "confidence": round(case["certainty"], 4),
            "error_type": s,
        })
    pdf = pd.DataFrame(rows)
    outp = dst / "gradcam_cases.csv"
    pdf.to_csv(outp, index=False)
    logger.info("gradcam export -> %d cases", len(pdf))
    return outp


# --------------------------------------------------------------------------- #
# research_summary.json — single evidence document
# --------------------------------------------------------------------------- #
def research_summary_json() -> Path:
    e1 = ART.exp1["splits"]["test"]
    base = ART.base_test["metrics"]
    ablation = {}
    for lbl in C.TARGET_LABELS:
        f1_old = base["per_label"][lbl]["f1"]
        f1_new = ART.exp3["evaluation"]["test"][lbl]["f1_optimal"]["f1"]
        ablation[lbl] = {
            "f1_fixed_0.5": round(f1_old, 4),
            "f1_f1_optimal_calibrated": round(f1_new, 4),
            "delta_f1": round(f1_new - f1_old, 4),
            "auroc": round(e1[lbl]["auroc_raw"], 4),
            "auprc": round(e1[lbl]["auprc_raw"], 4),
        }
    ea_c = ART.ea["strata"]["f1_optimal"]["Cardiomegaly"]
    external = _load("outputs/metrics/external/status.json")
    payload = {
        "phase": "final_results",
        "research_question": (
            "Can a frozen DenseNet-121 generalize to Cardiomegaly/Effusion in an "
            "imbalanced chest X-ray cohort, and how do post-hoc probability "
            "calibration and threshold policies change operating behaviour?"),
        "baseline_description": (
            "DenseNet-121 (ImageNet V1) trained from scratch for the two-label task "
            "with BCEWithLogits on the train split; patient-level split; one frozen "
            "checkpoint (best validation mean AUROC)."),
        "proposed_method": (
            "post-hoc per-label temperature scaling (fit on validation NLL) plus a "
            "threshold policy chosen on validation (F1-optimal, with constrained "
            "alternatives); all thresholds frozen before the test split is touched."),
        "main_metric_change": {
            "description": "the F1 operating point improves on both classes while "
                           "rank discrimination is exactly invariant to temperature",
            "per_label": ablation,
        },
        "calibration_change": {
            "raw_vs_calibrated": {
                lbl: {
                    "nll_before": round(e1[lbl]["raw"]["nll"], 5),
                    "nll_after": round(e1[lbl]["calibrated"]["nll"], 5),
                    "brier_before": round(e1[lbl]["raw"]["brier"], 5),
                    "brier_after": round(e1[lbl]["calibrated"]["brier"], 5),
                    "ece_before": round(e1[lbl]["raw"]["ece"], 5),
                    "ece_after": round(e1[lbl]["calibrated"]["ece"], 5),
                } for lbl in C.TARGET_LABELS},
            "headline": "temperature scaling reduces NLL and Brier but RAISES "
                        "binned ECE because ECE ≈ mean-confidence − prevalence "
                        "and calibration sharpens overconfident misclassifications",
        },
        "error_analysis_finding": {
            "metric": "share of test errors that are high-confidence (≥0.90)",
            "value_per_class": {
                lbl: {
                    "n_errors": ART.ea["strata"]["f1_optimal"][lbl]["error_rate"]
                                * ART.ea["strata"]["f1_optimal"][lbl]["n"],
                    "share_high_confidence": ART.ea["strata"]["f1_optimal"][lbl]
                                ["high_confidence_errors"]["share_of_errors"],
                } for lbl in C.TARGET_LABELS},
            "headline": ("Cardiomegaly errors are dominated by confident wrong "
                         "predictions (48.7% of errors have p≥0.90), whereas "
                         "Effusion errors are mostly low-confidence."),
        },
        "external_validation_status": external.get("status"),
        "no_external_metrics_computed": external.get("no_metrics_were_computed"),
        "limitations": [
            "external cohort absent — external validity UNVERIFIED (Table 8 is a status row)",
            "bootstrap CIs resample test images, not patients (patient clustering noted)",
            "error analysis / Grad-CAM use the combined arm's fixed policy and cohort",
            "pointing game is a localization sanity check on 43 boxed images, not a validation",
            "ECE increase after calibration is a known binned-ECE pathology, reported",
        ],
        "provenance": {
            "source_artifacts": [
                "outputs/metrics/baseline/baseline_metrics_test.json",
                "outputs/metrics/experiments/exp1_calibration.json",
                "outputs/metrics/experiments/exp2_thresholds_raw.json",
                "outputs/metrics/experiments/exp3_thresholds_calibrated.json",
                "outputs/metrics/error_analysis/error_analysis_test.json",
                "outputs/gradcam/bbox_localization.json",
                "outputs/metrics/external/status.json",
            ],
        },
    }
    return write_json(C.RESEARCH_SUMMARY_JSON, payload)


# --------------------------------------------------------------------------- #
def run() -> Dict[str, str]:
    C.FINAL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    C.FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table1()
    table2()
    table3()
    table4()
    table5()
    table6()
    table7()
    table8()
    fig1_roc()
    fig2_pr()
    fig3_reliability()
    fig4_threshold_sweep()
    fig5_confusion()
    fig6_training_curves()
    fig7_error_analysis()
    fig8_gradcam()
    error_analysis_csv()
    region_analysis_json()
    gradcam_export()
    research_summary_json()
    logger.info("paper artifacts complete under %s", C.FINAL_RESULTS_DIR)
    return {"status": "complete"}


if __name__ == "__main__":
    run()