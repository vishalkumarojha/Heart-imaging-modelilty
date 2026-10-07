"""Probability calibration for the frozen X-ray classifier.

Two families of quantity are produced here:

* **measurement** — how badly calibrated are the raw probabilities?
  `expected_calibration_error`, `brier_score`, `reliability_bins`,
  `calibration_report`.
* **post-hoc correction** — `TemperatureScaler`, a single scalar temperature per
  label fitted by maximising likelihood (minimising NLL) on **validation logits
  only**, then applied to any split.

Why temperature scaling and not Platt/Isotonic?
    It is a one-parameter, monotone map logit → logit/T, so it cannot change
    ranking: AUROC and AUPRC are *exactly* invariant (asserted in
    `tests/test_calibration.py`). Isotonic/Platt would need more data and can
    overfit 16 k validation rows; there is no evidence yet that richer
    calibration is warranted.

Hard rule enforced by the API, not by convention
    `TemperatureScaler.fit` records the split it was fitted on and refuses to be
    fitted on `test` / `external` splits — calibration must never see evaluation
    data (`docs/PROJECT_SCOPE.md` §4).

    from src.calibration import TemperatureScaler, calibration_report
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from . import config as C
from .reproducibility import write_json

FORBIDDEN_FIT_SPLITS = ("test", "external", "chexpert")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _as1d(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def _clip(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def _bin_edges(n_bins: int, strategy: str, conf: np.ndarray) -> np.ndarray:
    if strategy == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "equal_freq":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(conf, quantiles)
        edges[0], edges[-1] = 0.0, 1.0
        # guard against duplicate edges when scores are heavily tied
        edges = np.maximum.accumulate(edges)
        return edges
    raise ValueError(f"unknown binning strategy: {strategy!r}")


def reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = C.CALIBRATION_BINS,
    strategy: str = C.CALIBRATION_STRATEGY,
) -> Dict[str, np.ndarray]:
    """Confidence/accuracy per bin — the data behind a reliability diagram.

    Returns bin_edges (n_bins+1), bin_centers, mean_confidence, empirical_accuracy,
    count. Empty bins have NaN accuracy and count 0 (never silently 0.0).
    """
    y_true = _as1d(y_true)
    y_prob = _as1d(y_prob)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    conf = _clip(y_prob)
    edges = _bin_edges(n_bins, strategy, conf)
    idx = np.digitize(conf, edges[1:-1], right=False)  # 0 .. n_bins-1

    counts = np.bincount(idx, minlength=n_bins).astype(np.int64)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    acc_sum = np.bincount(idx, weights=y_true, minlength=n_bins)

    with np.errstate(invalid="ignore"):
        mean_conf = np.where(counts > 0, conf_sum / counts, np.nan)
        acc = np.where(counts > 0, acc_sum / counts, np.nan)

    return {
        "bin_edges": edges,
        "bin_centers": (edges[:-1] + edges[1:]) / 2.0,
        "mean_confidence": mean_conf,
        "empirical_accuracy": acc,
        "count": counts,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = C.CALIBRATION_BINS,
    strategy: str = C.CALIBRATION_STRATEGY,
) -> float:
    """ECE = Σ_b (n_b/N) · |accuracy_b − confidence_b| ∈ [0, 1].

    Empty bins contribute 0 (weight = 0). A model that outputs exactly the
    empirical frequency in every bin has ECE = 0; a model that outputs 1.0 for
    everything that is wrong has a large ECE.
    """
    b = reliability_bins(y_true, y_prob, n_bins, strategy)
    n = int(b["count"].sum())
    if n == 0:
        return float("nan")
    weights = b["count"] / n
    gaps = np.abs(b["empirical_accuracy"] - b["mean_confidence"])
    return float(np.nansum(weights * np.nan_to_num(gaps, nan=0.0)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of probabilities: mean((p − y)²) ∈ [0, 1]."""
    y_true, y_prob = _as1d(y_true), _as1d(y_prob)
    if y_true.shape != y_prob.shape:
        raise ValueError("shape mismatch")
    return float(np.mean((y_prob - y_true) ** 2))


def negative_log_likelihood(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Bernoulli NLL (natural log), averaged over rows."""
    p = _clip(_as1d(y_prob))
    y = _as1d(y_true)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def maximum_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray,
    n_bins: int = C.CALIBRATION_BINS,
    strategy: str = C.CALIBRATION_STRATEGY,
) -> float:
    """Worst single-bin gap (MCE) — reported alongside ECE, never instead of it."""
    b = reliability_bins(y_true, y_prob, n_bins, strategy)
    gaps = np.abs(b["empirical_accuracy"] - b["mean_confidence"])
    return float(np.nanmax(gaps)) if np.any(~np.isnan(gaps)) else float("nan")


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = C.CALIBRATION_BINS,
    strategy: str = C.CALIBRATION_STRATEGY,
) -> Dict[str, float]:
    """ECE + MCE + Brier + NLL + prevalence + mean confidence for one label."""
    y_true = _as1d(y_true)
    y_prob = _as1d(y_prob)
    return {
        "ece": expected_calibration_error(y_true, y_prob, n_bins, strategy),
        "mce": maximum_calibration_error(y_true, y_prob, n_bins, strategy),
        "brier": brier_score(y_true, y_prob),
        "nll": negative_log_likelihood(y_true, y_prob),
        "n": int(y_true.size),
        "prevalence": float(y_true.mean()) if y_true.size else float("nan"),
        "mean_confidence": float(_clip(y_prob).mean()) if y_prob.size else float("nan"),
        "n_bins": int(n_bins),
        "binning": strategy,
    }


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #
def _nll_for_temperature(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    return negative_log_likelihood(y, sigmoid(np.asarray(logits) / temperature))


def fit_temperature(
    logits: np.ndarray,
    y: np.ndarray,
    bounds: Sequence[float] = C.TEMPERATURE_BOUNDS,
) -> float:
    """Optimal scalar T > 0 minimising NLL of sigmoid(logit / T).

    One-dimensional bounded optimisation (`scipy.optimize.minimize_scalar`,
    Brent); if scipy is unavailable a golden-section search with the same bounds
    is used, so the function never depends on a package that is not in
    requirements.txt. Returns 1.0 for degenerate input (constant logits / no
    positives / no negatives), i.e. "leave the model alone".
    """
    logits = _as1d(logits)
    y = _as1d(y)
    if logits.shape != y.shape:
        raise ValueError("shape mismatch")
    if np.unique(y).size < 2 or np.allclose(logits, logits[0]):
        return 1.0

    lo, hi = float(bounds[0]), float(bounds[1])
    try:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(
            lambda t: _nll_for_temperature(logits, y, t),
            bounds=(lo, hi), method="bounded",
            options={"xatol": 1e-6},
        )
        if res.success and np.isfinite(res.fun):
            return float(res.x)
    except Exception:  # scipy missing / optimizer failure -> golden section
        pass
    return _golden_section(lambda t: _nll_for_temperature(logits, y, t), lo, hi)


def _golden_section(f, lo: float, hi: float, iters: int = 200) -> float:
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
        if abs(b - a) < 1e-6:
            break
    return float((a + b) / 2.0)


def fit_temperatures(
    logits: np.ndarray, y: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    bounds: Sequence[float] = C.TEMPERATURE_BOUNDS,
) -> np.ndarray:
    """Independent temperature per label for a (N, C) label matrix."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if logits.shape != y.shape:
        raise ValueError(f"shape mismatch: {logits.shape} vs {y.shape}")
    return np.array(
        [fit_temperature(logits[:, i], y[:, i], bounds) for i in range(logits.shape[1])]
    )


def apply_temperature(logits: np.ndarray, temperature: Union[float, np.ndarray]) -> np.ndarray:
    """logit → logit / T  (monotone ⇒ ranking preserved exactly)."""
    return np.asarray(logits, dtype=np.float64) / np.asarray(temperature, dtype=np.float64)


class TemperatureScaler:
    """Per-label temperature scaling fitted on validation only.

    Usage:
        scaler = TemperatureScaler(labels=["Cardiomegaly", "Effusion"])
        scaler.fit(val_logits, val_y, split="val")
        cal_logits = scaler.transform(test_logits)     # → sigmoid for probabilities
        scaler.save(path) / TemperatureScaler.load(path)
    """

    def __init__(self, labels: Sequence[str]):
        self.labels: List[str] = list(labels)
        self.temperatures_: Optional[np.ndarray] = None
        self.metadata_: Dict[str, object] = {}

    # -- fitting -----------------------------------------------------------
    def fit(
        self,
        logits: np.ndarray,
        y: np.ndarray,
        split: str = C.THRESHOLD_FIT_SPLIT,
        extra: Optional[Dict[str, object]] = None,
    ) -> "TemperatureScaler":
        split_l = str(split).lower()
        if any(bad in split_l for bad in FORBIDDEN_FIT_SPLITS):
            raise ValueError(
                f"refusing to fit calibration on split '{split}': calibration may "
                "only be fitted on validation (or training) predictions"
            )
        logits = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if logits.shape != y.shape or logits.shape[1] != len(self.labels):
            raise ValueError(
                f"expected (N, {len(self.labels)}) logits/labels, got "
                f"{logits.shape} / {y.shape}"
            )

        self.temperatures_ = fit_temperatures(logits, y, self.labels)
        nll_before = [
            negative_log_likelihood(y[:, i], sigmoid(logits[:, i]))
            for i in range(logits.shape[1])
        ]
        nll_after = [
            negative_log_likelihood(y[:, i], sigmoid(logits[:, i] / self.temperatures_[i]))
            for i in range(logits.shape[1])
        ]
        self.metadata_ = {
            "split": split,
            "n_rows": int(logits.shape[0]),
            "labels": self.labels,
            "temperatures": {l: float(t) for l, t in zip(self.labels, self.temperatures_)},
            "nll_before": {l: float(b) for l, b in zip(self.labels, nll_before)},
            "nll_after": {l: float(a) for l, a in zip(self.labels, nll_after)},
            "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bounds": list(C.TEMPERATURE_BOUNDS),
        }
        if extra:
            self.metadata_.update(extra)
        return self

    # -- applying ----------------------------------------------------------
    def _check(self):
        if self.temperatures_ is None:
            raise RuntimeError("TemperatureScaler has not been fitted/loaded")

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Calibrated logits (per-column temperature)."""
        self._check()
        logits = np.asarray(logits, dtype=np.float64)
        single = logits.ndim == 1
        if single:
            logits = logits.reshape(-1, 1)
        if logits.shape[1] != len(self.labels):
            raise ValueError("logit columns do not match labels")
        out = apply_temperature(logits, self.temperatures_)
        return out.ravel() if single else out

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(self.transform(logits))

    def state_dict(self) -> Dict[str, object]:
        self._check()
        return {
            "type": "temperature_scaling",
            "labels": self.labels,
            "temperatures": [float(t) for t in self.temperatures_],
            "metadata": self.metadata_,
        }

    def save(self, path: Path) -> Path:
        return write_json(Path(path), self.state_dict())

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        state = json.loads(Path(path).read_text())
        if state.get("type") != "temperature_scaling":
            raise ValueError(f"{path} is not a temperature-scaler state file")
        scaler = cls(state["labels"])
        scaler.temperatures_ = np.asarray(state["temperatures"], dtype=np.float64)
        scaler.metadata_ = state.get("metadata", {})
        return scaler

    def __repr__(self) -> str:  # pragma: no cover
        if self.temperatures_ is None:
            return "TemperatureScaler(unfitted)"
        pairs = ", ".join(f"{l}={t:.3f}" for l, t in zip(self.labels, self.temperatures_))
        return f"TemperatureScaler({pairs}, fit_on={self.metadata_.get('split', '?')})"
