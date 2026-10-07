"""IEEE-upgrade statistical pipeline runner (single entry point).

Stages run in dependency order; each stage is idempotent (re-running re-derives
the same artifacts from the frozen predictions and frozen val parameters):

   1  alternative_calibration   fit logistic scaler on val + logistic-space thresholds
   2  threshold_stability       validate fitter equivalence + bootstrap τ-per-policy
   3  bootstrap                 patient-level CIs + paired arm deltas (test)
   4  delong                    DeLong AUROC SE/CI per label
   5  tests                     paired permutation tests + Holm
   6  ece_sensitivity           ECE over calibration × bins × strategy
   7  extension                 arms E/F + decision-policy matrix
   8  prevalence_shift          simulated prevalence sensitivity
   9  statistical_report        consolidated statistical_report.json

    python -m src.statistical_upgrade --stage 1..9 --all
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

from .utils import setup_logging

logger = setup_logging()

STAGES: dict[int, tuple[str, Callable[[], object]]] = {
    1: ("alternative_calibration",
        lambda: _run("alternative_calibration", "fit_all")),
    2: ("threshold_stability",
        lambda: _run("threshold_stability", "run_threshold_stability")),
    3: ("bootstrap",
        lambda: _run("statistics.bootstrap", "run_patient_bootstrap")),
    4: ("delong",
        lambda: _run("statistics.delong", "run_delong")),
    5: ("tests",
        lambda: _run("statistics.tests", "run_paired_tests")),
    6: ("ece_sensitivity",
        lambda: _run("ece_sensitivity", "run_ece_sensitivity")),
    7: ("extension_experiments",
        lambda: _run("extension_experiments", "run_extension_arms") and
               _run("extension_experiments", "run_decision_policy_analysis")),
    8: ("prevalence_shift",
        lambda: _run("prevalence_shift", "run_prevalence_shift")),
    9: ("statistical_report",
        lambda: _run("statistical_report", "run_statistical_report")),
}


def _run(module: str, fn: str) -> object:
    import importlib
    m = importlib.import_module(f"src.{module}")
    return getattr(m, fn)()


def run_stage(number: int) -> None:
    if number not in STAGES:
        raise ValueError(f"unknown stage {number!r}; known: {sorted(STAGES)}")
    name, fn = STAGES[number]
    logger.info("=== stage %d/%d: %s ===", number, max(STAGES), name)
    t0 = time.perf_counter()
    fn()
    logger.info("=== stage %d (%s) done in %.1fs ===", number, name, time.perf_counter() - t0)


def run_all(first: int = 1, last: int = 9) -> None:
    for number in range(int(first), int(last) + 1):
        if number not in STAGES:
            continue
        run_stage(number)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the IEEE-upgrade statistical stages")
    p.add_argument("--stage", type=int, default=None,
                   help="run only this stage (1..9)")
    p.add_argument("--all", action="store_true",
                   help="run all stages in order")
    p.add_argument("--first", type=int, default=1)
    p.add_argument("--last", type=int, default=9)
    args = p.parse_args()
    if args.stage is not None:
        run_stage(args.stage)
    elif args.all:
        run_all(args.first, args.last)
    else:
        p.print_help()
        sys.exit(1)