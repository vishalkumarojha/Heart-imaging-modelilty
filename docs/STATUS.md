# Project status

> **`CORE_RESEARCH: COMPLETE` · `BASELINE: FROZEN` · `CALIBRATION: COMPLETE`
> · `THRESHOLD ANALYSIS: COMPLETE` · `ABLATION: COMPLETE` ·
> `ERROR ANALYSIS: COMPLETE` · `EXPLAINABILITY: COMPLETE` ·
> `EXTERNAL_VALIDATION: PENDING` · `PAPER_PREPARATION: COMPLETE / READY FOR
> AUTHOR REVIEW`**
>
> The core research pipeline is **frozen and verified**. Every headline number
> in the README/status documents is cross-checked against the underlying JSON/
> CSV artifact by `tests/test_consistency.py` and
> `outputs/final_results/verification_report.json` (currently **25/25 PASS**).
> The one open research item is external validation, which is a **separate
> question** from the completed core work — it stays `PENDING` until an
> external cohort is available.

## What is COMPLETE (verified against artifacts)

| Item | Status | Evidence |
|---|---|---|
| Frozen baseline DenseNet121 (best val AUROC epoch 8) | `FROZEN` | `baseline_manifest.json` sha256 `35965f610c8b…`; hash-checked by tests |
| Patient-level 70/15/15 split (109 312 imgs / 29 720 patients, overlap 0) | `VERIFIED` | `split_index.csv` ↔ `table_1_dataset.csv` |
| Temperature scaling (T = 1.187 / 1.398, fit on val NLL) | `FROZEN` | `temperature_scalers.json`, snapshot |
| 5 threshold policies (fixed, F1-opt, Youden, sens@0.90, prec@0.50) fit on val | `FROZEN` | `thresholds_*.json`, `fit_split = val` (test-enforced) |
| Test metrics (AUROC/AUPRC/F1/sens/spec) | `VERIFIED` | `baseline_metrics_test.json`, master table |
| Calibration report raw vs calibrated (NLL↓, Brier↓, ECE↑ explained) | `VERIFIED` | `exp1_calibration.json` |
| exp2 ≡ exp3 threshold equivalence (confusion matrices) | `VERIFIED` | tests + `comparison_note` |
| Error analysis (high-conf errors, strata, patients, cross-label) | `VERIFIED` | `error_analysis_test.json` |
| Grad-CAM case selection + bbox pointing (43 imgs, 21/43) | `VERIFIED` | `gradcam/bbox_localization.json` |
| Bootstrap confidence intervals (2000 iters, seed 42) | `VERIFIED` | `confidence_intervals.csv` |
| Paper tables 1–8, figures 1–8, exports, research summary | `VERIFIED` | `outputs/final_results/` |
| ECE definition (independent recompute, equal-width 15 bins) | `VERIFIED` | `test_consistency.py::TestECEImplementationIsStandard` |
| **Paper package** (evidence map 69/69, tables 01–09, figures 01–09, claim audit all-true, reproducibility manifest, prose drafts) | `VERIFIED` | `outputs/paper/`, `test_paper_package.py` |
| 193-test suite | `PASS` | `unittest discover -s tests` (~15 s) |

## What is PENDING (the only open question)

| Item | Status | Why |
|---|---|---|
| External validation (CheXpert cohort) | `PENDING` | no CheXpert data found at `data/raw/chexpert/`, `$CHEXPERT_ROOT`, `$CHEXPERT_DATA_DIR`, `/mnt/data/chexpert` or `/data/chexpert` |

The `src/external_eval` pipeline (loader, `u_zeroes` uncertainty policy, frozen
model + temperature + thresholds, no refit) is fully implemented and unit-
tested; dropping the official files in place and re-running
`python -m src.external_eval --split valid` produces real numbers with **zero
further code**. Until then `outputs/metrics/external/status.json` records
`no_metrics_were_computed: true` and nothing else is written.

## What is explicitly out of scope (legend `PLANNED`/`LEGACY`)

* Re-training or replacing the baseline architecture — frozen control, by design.
* Additional NIH findings beyond Cardiomegaly + Effusion.
* The ECHO / MRI / fusion / demo code — `LEGACY`, kept for history, hard-depends
  on the root `outputs/checkpoints/densenet121_best.pt` (do not delete).
* Multi-seed runs — out of scope at freeze time; single-seed point estimates
  are reported with bootstrap CIs, both documented.

## How to re-verify everything

```bash
.venv/bin/python -m unittest discover -s tests            # 193 tests, all green
.venv/bin/python -m src.research_freeze --no-tests        # refresh the snapshot
.venv/bin/python -m src.master_results                    # → verification_report.json PASS
.venv/bin/python -m src.uncertainty                       # → confidence_intervals.csv
.venv/bin/python -m src.paper_artifacts                   # → final_results/** (deterministic)
.venv/bin/python -m src.paper_build                       # → outputs/paper/** (deterministic)
```

Any change that alters the frozen artifacts (checkpoint, predictions,
temperature, thresholds, split) will fail the consistency tests — that is by
design.