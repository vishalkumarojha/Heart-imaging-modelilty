# Reproducibility (Phase 11)

How to re-derive every number in this repository from the frozen artifacts.
The rule throughout: **one checkpoint + one cached prediction table + one set
of val-fitted parameters → all results.** Nothing is recomputed twice, so no
two figures can disagree.

## 1. The immutable inputs

| Artifact | Path | SHA-256 (first 12) |
|---|---|---|
| Baseline checkpoint | `outputs/checkpoints/baseline/densenet121_best.pt` | `35965f610c8b` |
| Baseline last-epoch | `outputs/checkpoints/baseline/densenet121_last.pt` | `e5f6360d834a` |
| Frozen split | `data/processed/split_index.csv` | 109,312 rows, 29,720 patients, seed 42 |
| Checkpoint manifest | `outputs/checkpoints/baseline/baseline_manifest.json` | written once, read-only thereafter |

`outputs/checkpoints/densenet121_best.pt` (project root) is the *same weights*
under a legacy name; `demo_app.py` and `src/fusion_config.py` require it.

Baseline split statistics (from `baseline_metrics_test.json`, machine-written):

| Split | Images | Patients | Cardiomegaly + | Effusion + |
|---|---|---|---|---|
| train | 76,977 | 20,797 | 1,880 | 9,001 |
| val | 16,451 | 4,468 | 392 | 1,966 |
| test | 15,884 | 4,455 | 415 | 1,996 |
| **total** | **109,312** | **29,720** | **2,687** | **12,963** |

Patient overlap train∩val = train∩test = val∩test = **0**.

## 2. Environment of record

Captured inside every result JSON (`environment` block):

```
python 3.14.7 · Linux 7.2.5-100.fc43.x86_64 · x86_64
i5-11400H · NVIDIA GTX 1650 (3.63 GB, cc 7.5) · CUDA 13.0 · cuDNN 92400
torch 2.14.0+cu130 · torchvision 0.29.0+cu130 · numpy 2.5.3 · pandas 3.0.5
scikit-learn 1.9.0 · scipy 1.18.1 · albumentations 2.0.8 · Pillow 12.3.0
matplotlib 3.11.1 · tqdm 4.70.0
```

Use `docs/TECH_STACK.md` §versions to recreate it; `pip install -r
requirements.txt` inside a fresh venv is sufficient for Phases 1–7.

**Interpreter:** `.venv/bin/python` (system `python3` has no torch).
**Test runner:** `unittest` — pytest is not installed.
**Plot backend:** matplotlib `Agg` (no display required).

## 3. Reproduction ladder

Each rung depends only on the rungs above it.

```bash
# Rung 0 — integrity (read-only, seconds)
.venv/bin/python -m unittest discover -s tests        # 140 tests, ~15 s

# Rung 1 — baseline evaluation from the frozen checkpoint
.venv/bin/python -m src.baseline_eval --split test
.venv/bin/python -m src.baseline_eval --split val
# → outputs/metrics/baseline/baseline_metrics{,_test,_val}.{json,csv}

# Rung 2 — cached logits (the single source for every experiment)
.venv/bin/python -m src.inference --split test
.venv/bin/python -m src.inference --split val
# → outputs/predictions/raw/{split}__densenet121_best__35965f610c8b.csv
#   (already present; re-running reads the cache, it does not re-infer)

# Rung 3 — fit on VAL ONLY, then freeze
.venv/bin/python -m src.fit_parameters                # refits and overwrites
.venv/bin/python -m src.fit_parameters --check        # verifies without touching

# Rung 4 — experiments 1–4
.venv/bin/python -m src.experiments --exp all
# → outputs/metrics/experiments/*, outputs/predictions/{calibrated,thresholded}/
#   outputs/plots/{calibration,threshold}/

# Rung 5 — error analysis + Grad-CAM
.venv/bin/python -m src.error_analysis --split test
.venv/bin/python -m src.gradcam
# → outputs/metrics/error_analysis/*, outputs/gradcam/*, plots/error_analysis/

# Rung 6 — external (BLOCKED without data; writes status.json only)
.venv/bin/python -m src.external_eval --status
```

### Expected values (what a correct rerun must produce)

| Rung | Assertion |
|---|---|
| 1 | test AUROC Cardiomegaly **0.8972**, Effusion **0.8587**, macro **0.8780** (tolerance ≤5e-6 vs stored; see §4) |
| 1 | val macro AUROC **0.8678** |
| 2 | row count = split size (15,884 / 16,451); checkpoint SHA prefix `35965f610c8b` in the filename |
| 3 | T = **1.1865413532791096** (Cardiomegaly), **1.3979213336039338** (Effusion); thresholds fixed 0.5/0.5, f1_optimal 0.9021/0.7699, youden 0.1826/0.4488, sens@0.90 0.0519/0.3044, prec@0.50 0.9822/0.8475 |
| 4 | test ECE raw: **0.106143 / 0.207741**; calibrated: **0.116876 / 0.229156** (ECE *rises* — see §4) |
| 4 | test F1 @ f1_optimal: **0.3536 / 0.4866** (vs 0.2836 / 0.4676 @ 0.5) |
| 4 | exp3 confusion matrices identical to exp2's (asserted by test) |
| 5 | 16 case IDs exactly as in `outputs/gradcam/case_list.json` |
| 5 | bbox cohort n=**43**, pointing **21/43**, mean concentration ratio **2.60**, `ratio_above_1` = **38** |
| 6 | `status.json` with `BLOCKED — EXTERNAL DATASET NOT AVAILABLE` and **no** `external_eval_*` file |

Re-running rung 5 twice must produce byte-identical case lists (selection is
ordered by certainty then image index — no RNG anywhere in it).

## 4. The three known non-determinism sources (and why they do not bite)

1. **AMP fp16 inference.** Recomputing the test set can flip ~4 Effusion rows
   across the 0.5 threshold (values sit exactly on the fp16 quantum
   0.500000–0.500977 vs 0.499863–0.499950). Score difference ≤1.6e-3, AUROC
   difference ≤5e-6. **Mitigation:** every experiment reads the *cached* CSV,
   so this delta appears at most once, in rung 1, and is documented with image
   IDs in `docs/BASELINE.md` §6.
2. **GPU atomics / cuDNN autotuning.** Not exercised: Phases 4–7 beyond the
   baseline are pure numpy/scipy on cached arrays; Grad-CAM runs in eval mode
   with `torch.no_grad`-free single-image backward, where the case list and
   images are fixed. If you need bit-exact Grad-CAM across machines, run
   `--device cpu`.
3. **`Data_Entry` row order / duplicate filenames.** The split index is frozen
   to CSV at seed 42 and never regenerated during the research phases.

## 5. Provenance chain for a published number

Take any number in the README/PHASE_REPORTS and walk back:

```
README table
  → docs/PHASE_REPORTS.md (phase report, what it was computed from)
    → outputs/metrics/<phase>/*.json           (machine-written source of truth)
      → outputs/predictions/{calibrated,thresholded}/*.csv   (rows behind it)
        → outputs/predictions/raw/*__35965f610c8b.csv        (logits)
          → outputs/checkpoints/baseline/densenet121_best.pt  (sha256 35965f610c8b)
```

Each arrow is a file on disk; the tests in `tests/test_experiment_outputs.py`,
`tests/test_error_analysis.py` and `tests/test_gradcam.py` re-walk the chain
upwards from the CSVs and assert the published values still match.

## 6. What cannot be reproduced (stated, not hidden)

* **External generalisation** — no CheXpert data in the repository; the phase
  is `BLOCKED — EXTERNAL DATASET NOT AVAILABLE`, and the status file says
  `no_metrics_were_computed: true`.
* **The original training run's exact weights history** — only `*_best.pt`
  and `*_last.pt` are retained; per-epoch weights were never saved.
* **NIH official-split comparability** — `train_val_list.txt` /
  `test_list.txt` are absent, so our patient-level split differs from the
  published one and absolute numbers are not comparable to papers using it.
* **Five defective PNGs** (1 truncated in test, 1 truncated + 3 empty in
  train) — documented in `docs/REPOSITORY_AUDIT.md` §2a; the loader's
  substitution behaviour is part of the frozen control and deliberately not
  "fixed".
