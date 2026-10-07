"""Alternative calibration method: Platt-style logistic scaling on logits.

The baseline pipeline calibrates with a single per-label temperature
(`src/calibration.TemperatureScaler`).  The IEEE upgrade adds a second,
strictly richer calibration family to (a) test how sensitive the ECE/decision
conclusions are to the calibration choice and (b) feed the extension arms E and
F of the ablation.

Model
-----
    p = σ(a·z + b),   z = raw logit,  a > 0 enforced via a = exp(c).

`a > 0` makes the map strictly increasing, so AUROC / AUPRC are exactly
invariant under logistic calibration (asserted below on val and test).
Parameters (c, b) minimise the Bernoulli negative-log-likelihood on the
validation split only (`scipy.optimize.minimize`, L-BFGS-B, deterministic
starting point) — same data-governance rule as temperature scaling
(`FORBIDDEN_FIT_SPLITS` refusal is enforced at runtime).

Artifacts
---------
outputs/metrics/calibration/logistic_scalers.json        per-label (a, b, method)
outputs/metrics/thresholds/thresholds_logistic_val.json  policies on σ(a·z+b) probs

Run order: this module MUST run before `src.statistics.bootstrap` and
`src.extension_experiments` (they consume both files).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from . import config as C
from .calibration import FORBIDDEN_FIT_SPLITS, calibration_report, negative_log_likelihood, sigmoid
from .fit_parameters import _write_if_allowed, fit_thresholds, thresholds_to_json
from .metrics import auroc, auprc
from .reproducibility import write_json
from .utils import setup_logging
from .statistics.data import load_val_predictions
from .inference import as_arrays, label_names_from

logger = setup_logging()

METHOD = "platt_logistic"     # σ(a·z + b), a>0, monotone


class LogisticScaler:
    """Platt-style logistic calibration fitted on validation logits only."""

    def __init__(self, labels: Sequence[str]):
        self.labels = list(labels)
        self.params_: Dict[str, Dict[str, float]] = {}
        self.metadata_: Dict[str, object] = {}

    def _nll(self, ab: np.ndarray, i: int, logits: np.ndarray, y: np.ndarray) -> float:
        c, b = ab
        p = sigmoid(np.exp(float(c)) * logits + float(b))
        return negative_log_likelihood(y, p)

    def fit(self, logits: np.ndarray, y: np.ndarray, split: str = C.THRESHOLD_FIT_SPLIT,
            extra: Optional[Dict[str, object]] = None) -> "LogisticScaler":
        sl = str(split).lower()
        if any(bad in sl for bad in FORBIDDEN_FIT_SPLITS):
            raise ValueError(
                f"refusing to fit logistic calibration on split '{split}': "
                "calibration may only be fitted on validation (or training) predictions"
            )
        logits = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if logits.shape != y.shape or logits.shape[1] != len(self.labels):
            raise ValueError(f"shape mismatch: {logits.shape} vs labels {len(self.labels)}")

        raw_params: Dict[str, Dict[str, float]] = {}
        nll_before: Dict[str, float] = {}
        nll_after: Dict[str, float] = {}
        for i, label in enumerate(self.labels):
            z, yy = logits[:, i], y[:, i]
            if np.unique(yy).size < 2 or np.allclose(z, z[0]):
                a, b_val = 1.0, 0.0      # degenerate -> identity mapping
            else:
                res = minimize(
                    lambda ab: self._nll(ab, i, z, yy),
                    x0=np.array([0.0, 0.0]),          # a = exp(0) = 1, b = 0
                    method="L-BFGS-B",
                    options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
                )
                if not res.success:
                    logger.warning("LogisticScaler: L-BFGS-B did not converge for %s: %s",
                                   label, res.message)
                c, b_val = float(res.x[0]), float(res.x[1])
                a = float(np.exp(c))
            raw_params[label] = {"a": a, "b": b_val, "method": METHOD,
                                 "a_positive": a > 0}
            nll_before[label] = negative_log_likelihood(yy, sigmoid(z))
            nll_after[label] = negative_log_likelihood(yy, sigmoid(a * z + b_val))

        self.params_ = raw_params
        self.metadata_ = {
            "type": f"{METHOD} (Platt-style)", "split": split,
            "n_rows": int(logits.shape[0]), "labels": self.labels,
            "optimizer": "scipy L-BFGS-B (deterministic, a=exp(c)>0)",
            "nll_before": nll_before, "nll_after": nll_after,
            "parameters": raw_params,
        }
        if extra:
            self.metadata_.update(extra)
        return self

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        single = logits.ndim == 1
        if single:
            logits = logits.reshape(-1, 1)
        if logits.shape[1] != len(self.labels):
            raise ValueError("logit columns do not match labels")
        cols = [sigmoid(self.params_[l]["a"] * logits[:, i] + self.params_[l]["b"])
                for i, l in enumerate(self.labels)]
        out = np.column_stack(cols)
        return out.ravel() if single else out

    def state_dict(self) -> Dict[str, object]:
        return {"type": "logistic_scaling", "labels": self.labels,
                "method": METHOD, "parameters": self.params_,
                "metadata": self.metadata_}

    def save(self, path: Path) -> Path:
        return write_json(path, self.state_dict())

    @classmethod
    def load(cls, path: Path) -> "LogisticScaler":
        state = json.loads(Path(path).read_text())
        if state.get("type") != "logistic_scaling":
            raise ValueError(f"{path} is not a logistic-scaler state file")
        scaler = cls(state["labels"])
        scaler.params_ = state["parameters"]
        scaler.metadata_ = state.get("metadata", {})
        return scaler


def fit_all(refit: bool = False) -> Dict[str, object]:
    """Fit logistic calibrator on validation + thresholds in logistic space.

    Reuses the *existing* fit_parameters predicate so a non-refit run never
    clobbers artifacts.  Fit is on the frozen validation predictions only.
    """
    from .fit_parameters import ensure_predictions   # local: same CLI contract

    ckpt = C.BASELINE_CHECKPOINT_BEST
    preds = ensure_predictions(ckpt, "val")
    labels = label_names_from(preds)
    arrays = as_arrays(preds, labels)
    y, logits = arrays["y_true"], arrays["logits"]

    scaler = LogisticScaler(labels).fit(logits, y, split="val")
    log_probs = scaler.probabilities(logits)

    if not C.LOGISTIC_SCALERS_FILE.exists() or refit:
        scaler.save(C.LOGISTIC_SCALERS_FILE)
    th = fit_thresholds(y, log_probs, labels, split="val")
    _write_if_allowed(C.LOGISTIC_THRESHOLDS_FILE, thresholds_to_json(th), refit, "logistic")

    # monotone-invariance hygiene on BOTH val and test
    checks = {}
    for split in ("val", "test"):
        df = load_val_predictions() if split == "val" else _load_test()
        yy, zz = df[[f"true_{l}" for l in labels]].to_numpy(), df[[f"logit_{l}" for l in labels]].to_numpy()
        p_raw, p_log = sigmoid(zz), scaler.probabilities(zz)
        for i, l in enumerate(labels):
            checks[f"{split}/{l}/auroc_raw"] = auroc(yy[:, i], p_raw[:, i])
            checks[f"{split}/{l}/auroc_log"] = auroc(yy[:, i], p_log[:, i])
            checks[f"{split}/{l}/auprc_raw"] = auprc(yy[:, i], p_raw[:, i])
            checks[f"{split}/{l}/auprc_log"] = auprc(yy[:, i], p_log[:, i])
            assert abs(checks[f"{split}/{l}/auroc_raw"] - checks[f"{split}/{l}/auroc_log"]) < 1e-12
            assert abs(checks[f"{split}/{l}/auprc_raw"] - checks[f"{split}/{l}/auprc_log"]) < 1e-12

    report = {
        "status": "LOGISTIC CALIBRATION FITTED ON VALIDATION ONLY",
        "method": METHOD,
        "model": "p = σ(a·z + b), a = exp(c) > 0",
        "split": "val",
        "n_rows": int(len(preds)),
        "parameters": scaler.params_,
        "nll_before": scaler.metadata_["nll_before"],
        "nll_after": scaler.metadata_["nll_after"],
        "val_report": {l: calibration_report(y[:, i], log_probs[:, i]) for i, l in enumerate(labels)},
        "auroc_auprc_invariance_checks": checks,
        "thresholds_logistic_val": {p: {l: float(r["threshold"]) for l, r in m.items()}
                                    for p, m in th.items()},
    }
    out = C.CALIBRATION_METRICS_DIR / "logistic_calibration_report.json"
    write_json(out, report)
    logger.info("Logistic calibration report -> %s", out)
    logger.info("logistic thresholds -> %s", C.LOGISTIC_THRESHOLDS_FILE)
    return report


def _load_test():
    from .statistics.data import load_test_predictions
    return load_test_predictions()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fit logistic calibration on validation")
    p.add_argument("--refit", action="store_true")
    args = p.parse_args()
    fit_all(args.refit)