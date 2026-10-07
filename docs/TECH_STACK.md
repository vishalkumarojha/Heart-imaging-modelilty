# Tech stack (Phase 10)

What runs this project, at the versions actually used. Every line here was read
from the running `.venv` or from an environment capture stored in the result
JSONs (`environment` blocks written by `src/reproducibility.py`), not from
memory.

## Runtime

| Component | Version | Notes |
|---|---|---|
| Python | 3.14.7 | built with GCC 15.3.1 |
| OS | Fedora 42-ish (`Linux 7.2.5-100.fc43.x86_64`), glibc 2.42 | x86_64 |
| CPU | 11th Gen Intel i5-11400H @ 2.70 GHz | 12 logical cores |
| RAM | ~15 GB | |
| GPU | NVIDIA GTX 1650, 3.63 GB (compute 7.5) | single GPU |
| CUDA runtime | 13.0 (via torch wheel) | cuDNN 92400 |
| Interpreter | **`.venv/bin/python`** | system `python3` has no torch — always use the venv |

## Core libraries (installed, verified)

| Package | Version | Role in this project |
|---|---|---|
| torch | 2.14.0+cu130 | model, AMP inference, Grad-CAM backward pass |
| torchvision | 0.29.0+cu130 | DenseNet121 (ImageNet-1K V1 weights) |
| numpy | 2.5.3 | arrays everywhere |
| pandas | 3.0.5 | prediction tables, split index, result CSVs |
| scikit-learn | 1.9.0 | `roc_auc_score`, `average_precision_score`, ROC/PR curves (reference implementation our `src/metrics.py` is tested against) |
| scipy | 1.18.1 | `minimize_scalar` for temperature fitting (bounded); golden-section fallback if it fails |
| albumentations | 2.0.8 | evaluation transform (resize + ImageNet normalize); training augmentation lives in `src/dataset.py` |
| matplotlib | 3.11.1 | all figures — **Agg** backend, forced before pyplot import |
| Pillow | 12.3.0 | image decode |
| tqdm | 4.70.0 | progress bars for inference |

Installed via `requirements.txt` (now includes `scipy`, which Phase 4 needs).

## Present in the environment but NOT used by the research code

| Package | Version | Why it is not part of the research stack |
|---|---|---|
| opencv-python-headless | 5.0.0.93 | pulled in by albumentations; not imported directly |
| nibabel | 5.4.2 | legacy MRI phase only |
| gradio | 6.x | legacy demo only |

## Deliberately absent

| Not installed | Consequence |
|---|---|
| **pytest** | tests are plain `unittest` (`python -m unittest discover -s tests`) |
| **seaborn** | every figure is hand-built on matplotlib (`src/plots.py`) |
| jupyter | notebooks are not part of the workflow; everything is a module CLI |

Adding either would change nothing scientifically, but the tests and plots are
written against what exists — do not assume `pytest`/`seaborn` are available.

## Code map (research phases)

| Module | Responsibility |
|---|---|
| `src/config.py` | paths, split fractions, labels, Phase-4/5/6 settings — single source of truth |
| `src/reproducibility.py` | `sha256_file`, atomic `write_json` (NaN → null), env/split-stat capture |
| `src/metrics.py` | AUROC, AUPRC, binary + multilabel reports |
| `src/inference.py` | model load, cached logit/probability extraction (sha-keyed) |
| `src/baseline_eval.py` | frozen-baseline evaluation → immutable JSON/CSV + manifest |
| `src/calibration.py` | ECE/MCE/Brier/NLL, reliability bins, temperature scaling |
| `src/thresholds.py` | five threshold policies + val-only fitting guard |
| `src/fit_parameters.py` | fit and freeze temperatures + thresholds on val (`--refit/--check`) |
| `src/experiments.py` | experiments 1–4 (`--exp {1,2,3,4,all}`) |
| `src/plots.py` | all figures (reliability, bars, curves, sweeps, error analysis) |
| `src/error_analysis.py` | strata, high-confidence errors, concentration, case list |
| `src/gradcam.py` | Grad-CAM overlays + bbox localization cohort |
| `src/external_eval.py` | CheXpert pipeline (BLOCKED until data exists) |

## Legacy modules (kept, out of scope)

`echo_*`, `mri_*`, `fusion_*`, `demo_app.py`, `load_encoders.py` — the
multi-modality phases and the Gradio demo. They import `src/dataset.py`,
`src/model.py` and `src/utils.py` (which are shared and unchanged) and they
hard-depend on the root `outputs/checkpoints/densenet121_best.pt`.
See `docs/LEGACY_MODALITIES.md`.

## Hardware assumptions

* All research commands run on GPU or CPU (inference auto-selects CUDA when
  available; Grad-CAM needs a backward pass, ~0.3 s/case on the GTX 1650).
* Nothing in Phases 4–7 requires more than ~1.5 GB VRAM — inference is
  batch-32 AMP on 224×224 images.
* The full test suite (140 tests) runs in ~15 s and never loads the model or
  touches the GPU — it validates published artifacts and pure functions.
