# Legacy modalities — ECHO, MRI and fusion

**Status: LEGACY.** This document records the state of the three non-X-ray
components of the repository as they exist today. **No source file of theirs was
created, moved, renamed or edited during this migration** — the only changes made
to the project are additive (new `src/` modules, `docs/`, `outputs/metrics|predictions`)
and confined to the X-ray research line.

Everything below is **VERIFIED** unless tagged otherwise: values were read from
the repository's own artifacts (`outputs/checkpoints/*`, `outputs/logs/*`,
`PHASE2_ECHO_SUMMARY.md`, `PHASE3_MRI_SUMMARY.md`, `FUSION_SUMMARY.md`) and the
files listed were confirmed to exist on disk.

---

## 1. Why they are legacy, not deleted

* They are **independent of the X-ray pipeline**: `src/echo_*.py` and
  `src/mri_*.py` import only `src.utils` (shared helpers); neither imports
  `src/config.py`, `src/dataset.py` or `src/model.py`.
* The **fusion code hard-codes the root X-ray checkpoint path**
  (`src/fusion_config.py: XRAY_CKPT` and `demo_app.py: CKPT["xray"]` →
  `outputs/checkpoints/densenet121_best.pt`). Moving or deleting that file
  breaks them, which is precisely why the baseline freeze was implemented as a
  *copy* (`outputs/checkpoints/baseline/`) instead of a move (risk R4).
* They carry the repository's Phase 2–4 history and the demo app; deleting them
  would destroy work that is complete and reproducible, in exchange for nothing
  the research questions in `docs/PROJECT_SCOPE.md` need.

## 2. ECHO — `src/echo_{config,dataset,model,train,evaluate,explore}.py`

| Item | Value |
|---|---|
| Dataset | EchoNet-Dynamic (`data/raw/echonet/EchoNet-Dynamic/`), videos + CSVs |
| Task | 3-class ejection-fraction bucket: <40 / 40–54 / ≥55 % |
| Model | ResNet18 per frame → 1-layer BiLSTM over 16 frames → 1024-d `EchoEncoder` + `Linear(1024, 3)` |
| Split | official `Split` column — 7,465 train / 1,288 val / 1,277 test videos |
| Training | inverse-frequency-weighted CE, batch 8, AMP, 3 frozen + 6 unfrozen epochs |
| Best result | macro AUROC **0.802** (test, 1,277 videos) |
| Checkpoints | `outputs/checkpoints/echo/echo_cnn_lstm_{best,last}.pt` **(present)** |
| Logs | `outputs/logs/echo/` **(present)** |
| Dependency file | `requirements_echo.txt` (adds opencv-python-headless) |
| X-ray interaction | none (imports `src.utils` only) |

**State: complete and working; untouched by this migration.**

## 3. MRI — `src/mri_{config,dataset,model,train,evaluate,explore}.py`

| Item | Value |
|---|---|
| Dataset | ACDC (`data/raw/acdc/training/`), 100 patients |
| Task | 5-class diagnosis: DCM / HCM / MINF / NOR / RV |
| Model | ResNet18 per slice over `[ED, ES, ED−ES]` channels → BiLSTM over 10 slices → 1024-d `MriEncoder` + Dropout/Linear head |
| Split | stratified patient-level 70/15/15 (70/15/15 patients) |
| Training | 25 epochs, **CNN frozen throughout**, lr 3e-4, batch 8 |
| Best result | test macro AUROC **0.706**, accuracy 6/15 (small-data limited) |
| Checkpoints | `outputs/checkpoints/mri/mri_resnet18_bilstm_{best,last}.pt` **(present)** |
| Logs | `outputs/logs/mri/`, caches `data/processed/mri/` **(present)** |
| Dependency file | `requirements_mri.txt` (adds nibabel) |
| X-ray interaction | none (imports `src.utils` only) |

**State: complete but weak (100 patients); untouched by this migration.**

## 4. Fusion — `src/fusion_{config,model,train,demo}.py`, `src/load_encoders.py`

| Item | Value |
|---|---|
| Design | frozen Phase 1–3 encoders → per-modality LayerNorm → learned missing-modality token → concat 3072-d → MLP(512) → three task heads |
| Cohort | **there is no paired tri-modal cohort** — training/validation is single-modality-present only; `fusion_demo.py` demonstrates explicitly labelled *synthetic* embedding combinations (`SYNTHETIC_demo_predictions.csv`) |
| Validated numbers | `outputs/logs/fusion/single_modality_validation.csv`: xray 0.8651 (fusion pathway) vs 0.8780 (standalone, Δ −0.0129); echo 0.8059 vs 0.8020 (Δ +0.0039); mri 0.6278 vs 0.7060 (Δ −0.0782) |
| Checkpoints | `outputs/checkpoints/fusion/fusion_{best,last}.pt` **(present)** |
| Logs | `outputs/logs/fusion/` **(present)** |
| Coupling | **depends on the root X-ray checkpoint** (see §1) — this is the single hard constraint the migration works around |
| X-ray interaction | reads `outputs/checkpoints/densenet121_best.pt`; otherwise independent |

**State: complete as a demonstration; untouched by this migration.**

## 5. Demo application

`demo_app.py` (Gradio, 4 tabs: X-ray / ECHO / MRI / Fusion) plus `DEMO_GUIDE.md`
and `demo_samples/` (3 X-ray + 3 echo + 6 MRI samples).

* Loads `demo_app.py:CKPT = {"xray": outputs/checkpoints/densenet121_best.pt, ...}`
  → same hard-coded root path as fusion.
* Dependency file: `requirements_demo.txt` (gradio).
* Two **uncommitted** edits exist on `main` (a `GRADIO_SERVER_PORT` environment
  variable override); they were left untouched by this migration — see
  `docs/REPOSITORY_AUDIT.md` §1 (risk R12).

**State: LEGACY, functional, untouched.**

## 6. Effect of the X-ray research line on legacy components

| Risk | Mitigation actually implemented (Phase 2) |
|---|---|
| Freezing/moving the X-ray checkpoint breaks fusion + demo | baseline stored as a **copy** in `outputs/checkpoints/baseline/`; root `outputs/checkpoints/densenet121_{best,last}.pt` still present and byte-identical |
| Shared `src/` package changes break legacy imports | new modules only *added* (`src/{metrics,inference,baseline_eval,reproducibility}.py`); `src/config.py` extended with new path constants only — no existing constant changed |
| Output-path collisions (`R3`) | legacy `outputs/logs/*` are read-only for this project; new work writes to `outputs/{metrics,predictions,plots}` |
| Tests could fail because they touch legacy code | tests are scoped to the X-ray research modules; no legacy module is imported by them |

## 7. What "LEGACY" means for future work

* They continue to run with their own requirement files
  (`requirements_{echo,mri,demo}.txt`).
* They are **not** extended, refactored, re-trained or re-validated by this
  project; their numbers are reported as-is from their Phase 2–4 artifacts.
* Any experiment that would require modifying them (e.g. a fusion-aware
  calibration study) is **out of scope** and would need a new, explicitly
  approved work item.
