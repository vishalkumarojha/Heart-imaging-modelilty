# QA report (Phase 12)

Evidence, not intent. Every claim below is reproducible with the command in
its own row.

## 1. Test suite

```bash
.venv/bin/python -m unittest discover -s tests     # 140 tests, ~15 s, all green
```

| File | Tests | What it protects |
|---|---|---|
| `test_baseline_artifacts.py` | 20 | frozen checkpoint hashes unchanged, root checkpoint still present for the legacy demo, split integrity (no patient in two splits), the documented 5-defect PNG set is unchanged, cached CSV ↔ checkpoint SHA linkage, `prob == sigmoid(logit)` |
| `test_calibration.py` | 21 | ECE/Brier/NLL vs hand-computed values, bin edge cases (empty bins, single bin, perfect/uniform), temperature fit recovers known T on synthetic logits, **`TemperatureScaler` refuses `test`/`external` fit splits** |
| `test_thresholds.py` | 18 | all five policies against brute-force search, monotonicity (higher sensitivity target ⇒ lower τ), **`fit_thresholds` refuses eval splits**, `apply_thresholds` misalignment guard |
| `test_fit_parameters.py` | 7 | frozen JSONs exist, temperatures within bounds, thresholds sorted/finite, `--check` mode does not rewrite files |
| `test_metrics.py` | 12 | AUROC/AUPRC agree with sklearn on fixed arrays, degenerate inputs (all-positive/all-negative) raise or return the documented value |
| `test_experiment_outputs.py` | 17 | published ECE/Brier/confusion **recomputed from the stored prediction CSVs**, exp2 ≡ exp3 operating points, fixed@0.5 == baseline numbers, AUROC invariant to temperature, exp4 binning/target/bootstrap coherence, summary CSV row count |
| `test_error_analysis.py` | 18 | strata sum to n and to the positives count, strata recomputed from predictions, confidence bins partition the cohort, errors concentrate in high-confidence bins, case list re-selection byte-identical, case images exist |
| `test_gradcam.py` | 14 | `bbox_localization` math on synthetic heat maps (inside/outside/scaling/clipping/uniform), bbox reference covers both labels, overlays exist and are non-trivial, case IDs match the frozen case list, bbox cohort statistics recomputed, **cohort == the full boxed subset (no selection)** |
| `test_external_eval.py` | 13 | availability detection, **`run()` writes only `status.json` with `no_metrics_were_computed`, no `external_eval_*` / prediction file exists**, uncertainty policy (u_zeroes/u_ones/guard), synthetic CheXpert layout loader (label mapping, missing images ⇒ `FileNotFoundError`, missing column ⇒ `ValueError`) |
| **Total** | **140** | |

Design notes:

* Tests run against **published artifacts**, not fixtures — they fail if a
  result file is edited by hand.
* The honesty tests are negative assertions (`assertEqual(artifacts, [])`),
  so "no data ⇒ no result" is enforced, not documented.
* The suite never needs the GPU and never re-infers the full test set
  (~15 s total).

## 2. Data integrity

| Check | Method | Result |
|---|---|---|
| Image files present | recursive scan of `images_*` | **109,312** unique files (Data_Entry rows 112,120; 2,808 absent ≈ 2.5 %, `images_012` partial) |
| PNG structural validity | `IEND` trailer check on all 109,312 | **5 defects**: `00029705_000` truncated (test), `00029717_001` truncated (train), `00029718/19/20_000` zero-byte (train) |
| Defects unchanged | `test_known_defective_files_are_still_the_documented_set` | pass — the set is frozen as documented |
| Test-split defect count | `test_test_split_contains_exactly_one_defective_image` | pass (1 of the 5) |
| Patient leakage | split index cross-tab | train∩val = train∩test = val∩test = **0** patients |
| Split frozen | `data/processed/split_index.csv` | 109,312 rows, 76,977 / 16,451 / 15,884 |
| Duplicate filenames across `images_*` folders | path scan | 0 duplicates at scan time (handled by the scan's "keep last" rule if any appear) |

Consequences of the 5 defects (documented, deliberately not fixed — see
`docs/REPOSITORY_AUDIT.md` §2a): the dataset loader substitutes another valid
image for the truncated test PNG, so **1 test row's image ≠ its filename**; this
is part of the frozen control.

## 3. Guard rails (violation ⇒ test failure or runtime error)

| Rule | Enforcement point | Test |
|---|---|---|
| Temperature fitted on val only | `TemperatureScaler.fit(split=...)` raises `ValueError` for `test`/`external` | `test_calibration.py` |
| Thresholds fitted on val only | `fit_thresholds(split=...)` raises `ValueError` for `test`/`external` | `test_thresholds.py` |
| Baseline checkpoint never overwritten | hash assertions on `densenet121_best.pt` / `_last.pt` | `test_baseline_artifacts.py` |
| Root checkpoint kept for legacy demo | existence + hash | `test_baseline_artifacts.py` |
| Cache tied to checkpoint | filename contains `sha256[:12]` | `test_baseline_artifacts.py` |
| Prediction row count == split size | runtime guard in `predict_split` | `test_baseline_artifacts.py` |
| No external metrics without data | `run()` short-circuits to `status.json` | `test_external_eval.py` (negative file assertions) |
| NaN never leaks into JSON | `write_json` maps NaN/Inf → null | exercised by every result writer |
| Case list deterministic | re-selection equality | `test_error_analysis.py` |
| Grad-CAM cohort not cherry-picked | cohort == full boxed subset | `test_gradcam.py` |

## 4. Result cross-checks (published number ⇒ recomputation)

| Published | Source file | Recomputed by |
|---|---|---|
| AUROC/AUPRC/confusion per split | `outputs/metrics/baseline/*.json` | `test_baseline_artifacts.py` + sklearn reference in `test_metrics.py` |
| ECE / Brier / NLL, raw vs calibrated | `outputs/metrics/experiments/exp1_calibration.json` | `test_experiment_outputs.py` from `outputs/predictions/calibrated/*.csv` |
| Per-policy precision/recall/specificity/F1 | `exp2/exp3_thresholds*.json` | `test_experiment_outputs.py` (confusion recomputed, exp2 ≡ exp3 asserted) |
| ECE bin-invariance, target sweeps, bootstrap | `exp4_robustness.json` | coherence assertions in `test_experiment_outputs.py` |
| Strata / high-confidence errors / concentration | `outputs/metrics/error_analysis/error_analysis_test.json` | `test_error_analysis.py` (counts recomputed) |
| Grad-CAM bbox statistics | `outputs/gradcam/bbox_localization.json` | `test_gradcam.py` (recomputed from stored cases) |
| External BLOCKED status | `outputs/metrics/external/status.json` | `test_external_eval.py` |

## 5. Known open issues (not bugs)

1. **ECE rises after temperature scaling** (test: 0.106→0.117, 0.208→0.229).
   Expected: temperature scaling minimises NLL, not binned ECE; with every bin
   over-confident the two objectives disagree. Recorded as a finding.
2. **exp2 and exp3 produce identical confusion matrices.** Provable monotone
   equivalence; asserted by a test so a future change that *does* separate them
   is noticed.
3. **AMP fp16 ±4-row threshold nondeterminism** in baseline inference
   (≤5e-6 AUROC). Documented; mitigated by the single cached table.
4. **Effusion Grad-CAM pointing game = 2/19** despite concentration ratio 2.46.
   Reported as measured; the metric and its disagreement with the mass ratio
   are both published rather than choosing the flattering one.
5. **No external validation.** `BLOCKED — EXTERNAL DATASET NOT AVAILABLE`.

## 6. Lint / static checks

* `python -m py_compile src/*.py tests/*.py` clean (compile gate used after
  every edit).
* No linter (ruff/flake8) and no formatter (black) are installed in this
  environment; style is matched by hand to the existing modules.
* No type checker (mypy/pyright) installed; annotations are written but not
  machine-verified.
