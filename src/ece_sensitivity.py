"""Calibration (ECE) sensitivity to the binning grid and calibration method.

Sensitivity analysis: ECE is only meaningful relative to a binning choice
(Naeini, Liaw — ECE with fixed bins hides miscalibration).  This module wallpapers
the ECE over
    calibration method × {raw, temperature, logistic}    (CALIBRATION_COMPARISON_METHODS)
    bin count        × {10, 15, 20}                      (ECE_SENSITIVITY_BINS)
    binning strategy × {equal_width, equal_freq}
on the frozen test predictions, so the paper can (a) report a range instead of a
single number and (b) show that the ordering raw ≻? calibrated is (or is not)
robust to the grid.

Artifacts
---------
outputs/metrics/ece_sensitivity/ece_sensitivity.json
outputs/metrics/ece_sensitivity/ece_sensitivity.csv
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from . import config as C
from .calibration import expected_calibration_error
from .reproducibility import write_json
from .utils import setup_logging
from .statistics.data import (load_test_predictions, variant_probs)

logger = setup_logging()


def run_ece_sensitivity() -> dict:
    df = load_test_predictions()
    rows: List[dict] = []
    defaults = {l: {} for l in C.TARGET_LABELS}
    for label in C.TARGET_LABELS:
        y = df[f"true_{label}"].to_numpy(dtype=np.float64)
        for method in C.CALIBRATION_COMPARISON_METHODS:
            p = variant_probs(df, label, method)
            for nbins in C.ECE_SENSITIVITY_BINS:
                for strategy in ("equal_width", "equal_freq"):
                    ece = float(expected_calibration_error(y, p, nbins, strategy))
                    rows.append({"class": label, "calibration": method,
                                 "n_bins": int(nbins), "binning": strategy,
                                 "ece": round(ece, 6)})
                    if method == "calibrated" and nbins == C.CALIBRATION_BINS and strategy == C.CALIBRATION_STRATEGY:
                        defaults[label][method] = ece
                    if method in ("raw", "logistic") and nbins == C.CALIBRATION_BINS and strategy == C.CALIBRATION_STRATEGY:
                        defaults[label][method] = ece

    df_out = pd.DataFrame(rows)
    df_out.to_csv(C.ECE_SENSITIVITY_DIR / "ece_sensitivity.csv", index=False)

    payload = {
        "split": "test",
        "bins": list(C.ECE_SENSITIVITY_BINS),
        "strategies": ["equal_width", "equal_freq"],
        "methods": list(C.CALIBRATION_COMPARISON_METHODS),
        "reference_binning": {"n_bins": C.CALIBRATION_BINS, "strategy": C.CALIBRATION_STRATEGY},
        "ece_at_reference_binning": defaults,
        "rows": rows,
        "n_rows": len(rows),
    }
    write_json(C.ECE_SENSITIVITY_JSON, payload)
    logger.info("ece_sensitivity.json/.csv -> %d cells", len(rows))
    return payload


if __name__ == "__main__":
    run_ece_sensitivity()