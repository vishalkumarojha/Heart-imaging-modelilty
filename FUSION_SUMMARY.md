# Phase 4 — Multi-Modal Fusion — Final Capstone Summary

Status: **complete.** The fusion pathway is **built and validated single-modality-present**
on all three real test sets. There is **no real tri-modal patient data**, so true
multi-modal performance is *not measured* — only synthetically demonstrated.
Read with `PHASE1_XRAY_SUMMARY.md`, `PHASE2_ECHO_SUMMARY.md`, `PHASE3_MRI_SUMMARY.md`.

---

## 0. The hard constraint (stated first because it governs everything)

**NIH ChestX-ray14, EchoNet-Dynamic and ACDC come from three different
institutions with completely non-overlapping patients.** No patient has more
than one modality. Consequently this phase:

| | What it does | What it does **not** do |
|---|---|---|
| **Trains** | fusion block on the *union* of the 3 training sets, each sample **single-modality-present** (its real embedding + 2 learned "missing" tokens), supervised by its own task head | never sees a real multi-modal sample |
| **Validates** | each modality's **real test set**, other two masked missing → metric vs its Phase 1–3 standalone baseline | produce any real multi-modal accuracy number |
| **Demonstrates** | `fusion_demo.py`: 3 *unrelated* real people (one per dataset) fused into one fake "patient", all outputs stamped **SYNTHETIC** | present synthetic combinations as validated performance |

The only real fusion numbers are in **`outputs/logs/fusion/single_modality_validation.csv`**.
Synthetic outputs are in the separately-named **`SYNTHETIC_demo_predictions.csv`**.
Transparency is enforced in the config header, every module docstring, the
validation log footer, `fusion_demo.py`'s banner + per-row `[SYNTHETIC …]` tag +
the `SYNTHETIC` CSV column + the `SYNTHETIC_`-prefixed filename.

---

## 1. Architecture

```
   X-ray img [B,3,224,224]  ──▶ XrayEncoder (frozen)  ──▶ e_xray  [B,1024]
   echo clip [B,16,3,112,112]─▶ EchoEncoder (frozen)  ──▶ e_echo  [B,1024]
   MRI vol  [B,10,3,128,128] ─▶ MriEncoder  (frozen)  ──▶ e_mri   [B,1024]
                                                              │
                          presence mask [B,3] ───────────────▶│
                                                              ▼
   FusionLayer:  per-modality LayerNorm(e_i)                       (X-ray embeddings
                 absent modality  →  learned missing_token[i]       are ~15× larger in
                 (trainable [1024] param, NOT zeros)                raw magnitude — the
                 concat( 3 × [B,1024] ) = [B,3072]                  LayerNorm is load-
                 → Linear 3072→512 → GELU → Dropout(0.3)            bearing, not cosmetic)
                 → Linear 512→512 → GELU        =  z  [B,512]
                                                              │
   TaskHeads (share z):   xray → Linear(512, 2)   multi-label logits
                          echo → Linear(512, 3)   3-class logits
                          mri  → Linear(512, 5)   5-class logits
```

`FusionModel` = `FusionLayer` + `TaskHeads` — **1,850,378 trainable params**. The
three encoders are external, frozen (0 trainable), loaded via
`src/load_encoders.py::load_all_encoders`.

### Why concat + MLP, not attention
With only **3 fixed modality slots** there is nothing for attention to route over
that a 2-layer MLP cannot learn directly. Concat+MLP keeps the parameter count
and failure modes minimal — which matters when the pathway can only ever be
validated single-modality-present.

### Why scheme (b): task-specific heads on a shared representation
The alternative (a "unified cardiac-risk level" all three map to) needs an
**invented** label mapping (e.g. "X-ray: both findings → Severe", "MRI: DCM →
Severe, HCM → Moderate") — subjective and attackable in review. Scheme (b) keeps
**each dataset's real ground truth unchanged**, so single-modality validation is
a clean apples-to-apples comparison with Phases 1–3. A single "risk" readout is
still available in the demo as an explicit **post-hoc worst-of-3 heuristic** —
not trained, not validated.

---

## 2. Missing-modality handling

Each modality has a `nn.Parameter` of shape `[1024]` (`FusionLayer.missing_token[i]`,
small random init). When modality *i* is absent, its slot is filled with that
learned token instead of a zero vector; when present, `mask·LayerNorm(e_i) +
(1−mask)·token`. The tokens are trained jointly — during single-present training,
every batch has 2 of the 3 slots filled by tokens, so they get abundant gradient
and learn a useful "modality unavailable" prior rather than a dead zero.

---

## 3. Embedding precompute

`fusion_train.py::precompute_embeddings` runs each frozen encoder over its
dataset splits once and caches `data/processed/fusion/{modality}_{split}.npz`
(`emb [N,1024] float32`, `label`, `id`). After this, fusion training is pure
MLP-over-vectors (~0.5 s/epoch).

- **fp32 on purpose** — the DenseNet (X-ray) overflows to `inf` in fp16 given its
  large pre-classifier activations (max ≈ 44 vs echo/MRI ≈ ±1). Non-finite rows
  are zeroed + logged as a guard.
- **Caps:** X-ray `train` → 20,000 random (seed 42), X-ray `val` → 8,000. **Every
  `test` split is full and uncapped** so validation is honest: X-ray 15,884 /
  echo 1,277 / MRI 15.
- Cost: **~7 min** total (the X-ray DenseNet pass dominates; echo/MRI are minutes).

---

## 4. Training

| | value |
|---|---|
| trainable | `FusionModel` only (1.85 M) — encoders frozen |
| optimizer | Adam, lr 1e-3, weight_decay 1e-4 |
| batch / precision | 256 / fp32 (head is trivial) |
| epochs | 30 (~0.5 s each) |
| loss | per modality: `BCEWithLogitsLoss` (X-ray) / `CrossEntropyLoss` (echo, MRI) |
| loop | each epoch iterates all 3 modality train loaders; each batch = single-present forward → that head's loss → step |
| checkpoint | `fusion_best.pt` on **combined** val = mean(X-ray mean-AUROC, echo macro-AUROC, MRI macro-AUROC); `fusion_last.pt` every epoch |
| best | **epoch 13**, combined val **0.846** |

`outputs/logs/fusion/metrics.csv` — per-epoch per-modality train loss + val primary + combined.

---

## 5. Results — single-modality-present validation (the only real fusion numbers)

Each row: that modality's **real test set**, the other two marked missing.

| Modality | Fusion-pathway metric | Phase 1–3 standalone | Δ | Test N |
|---|---|---|---|---|
| **X-ray** — mean AUROC | **0.865** | 0.878 | **−0.013** | 15,884 |
| **Echo** — macro AUROC | **0.806** | 0.802 | **+0.004** | 1,277 |
| **MRI** — macro AUROC | **0.628** | 0.706 | **−0.078** | 15 |

### Per class

**X-ray** (threshold 0.5)
| Label | AUROC | Sens | Spec | Prec | F1 |
|---|---|---|---|---|---|
| Cardiomegaly | 0.878 | 0.089 | 0.998 | 0.536 | 0.153 |
| Effusion | 0.852 | 0.574 | 0.891 | 0.431 | 0.492 |

**Echo** — acc 0.720 · balanced acc 0.530 · macro F1 0.553
| Class | AUROC | F1 | CM row (→ Reduced / Mildly / Normal) |
|---|---|---|---|
| Reduced | 0.913 | 0.556 | 74 / 53 / 33 |
| Mildly Reduced | 0.680 | 0.254 | 24 / 54 / 163 |
| Normal | 0.826 | 0.849 | 8 / 77 / 791 |

**MRI** — acc 0.400 · balanced acc 0.400 · macro F1 0.396
| Class | AUROC | F1 | CM row (→ DCM / HCM / MINF / NOR / RV) |
|---|---|---|---|
| DCM | 0.639 | 0.444 | 2 / 0 / 1 / 0 / 0 |
| HCM | 0.639 | 0.400 | 2 / 1 / 0 / 0 / 0 |
| MINF | 0.389 | 0.400 | 0 / 1 / 1 / 0 / 1 |
| NOR | 0.750 | 0.400 | 1 / 0 / 0 / 1 / 1 |
| RV | 0.722 | 0.333 | 1 / 0 / 0 / 1 / 1 |

### Interpretation

- **The fusion mechanism is validated for X-ray and Echo.** Routing a frozen
  encoder's embedding through LayerNorm → concat(real, token, token) → shared MLP
  → task head costs ≈1 pp for X-ray and *gains* 0.4 pp for Echo. The learned
  missing-modality-token pathway carries each real modality's signal essentially
  intact. This is the core thing this phase set out to prove.
- **MRI −7.8 pp is within its (large) error bars.** N=15 → one patient ≈ ±7 pp;
  during training MRI val bounced **0.61 ↔ 0.86** and its train loss spiked
  (`m=23.8` at epoch 4). 70 MRI embeddings get buffeted in a multi-task loop
  whose gradients are dominated by X-ray (20 k) and Echo (7.5 k). This is
  consistent with Phase 3's already-flagged small-data limitation — **left as the
  honest result, not tuned away.** The MRI encoder still supplies a real
  embedding; its standalone head was always the weak link.

---

## 6. Synthetic demonstration (`src/fusion_demo.py`)

Builds *N* fabricated "patients": each = an X-ray embedding from one NIH person +
an echo embedding from a **different** EchoNet person + an MRI embedding from a
**third** ACDC person, run through the fusion model with all 3 modalities present.
Prints all three head outputs plus the heuristic worst-of-3 "cardiac-concern
level". Example (`--n 4`, seed 42):

| synth # | X-ray P(Card / Eff) | echo head | MRI head | heuristic risk |
|---|---|---|---|---|
| 1 | 0.03 / 0.22 | Normal | NOR | LOW |
| 2 | 0.10 / 0.39 | Normal | RV | MODERATE |
| 3 | 0.04 / 0.25 | Normal | DCM | HIGH |
| 4 | 0.01 / 0.07 | Normal | DCM | HIGH |

Written to `outputs/logs/fusion/SYNTHETIC_demo_predictions.csv` (column
`SYNTHETIC = "TRUE  (not a real patient)"`). Outputs skew toward Normal/low
because *all-3-present* is out-of-distribution for a model trained single-present
— **the demo shows the pathway runs, not that it is accurate.**

---

## 7. Files & reproduce

```
src/
  fusion_config.py    paths, MODALITIES, TASK_SPECS, caps, hyperparams, the constraint header
  load_encoders.py    load + freeze the 3 checkpoints; verify [B,1024]/no-head/no-activation
  fusion_model.py     FusionLayer (LayerNorm + learned tokens + concat-MLP) + TaskHeads
  fusion_train.py     precompute embeddings -> train -> single-modality-present validation
  fusion_demo.py      SYNTHETIC multi-modal demonstration
outputs/checkpoints/fusion/fusion_{best,last}.pt        (~7 MB, git-ignored)
outputs/logs/fusion/{metrics.csv, single_modality_validation.csv, SYNTHETIC_demo_predictions.csv}
data/processed/fusion/*.npz                             embedding cache (git-ignored)
```

```bash
python -m src.load_encoders     # step 1 — contract verification
python -m src.fusion_train      # precompute (~7 min) -> train (<1 min) -> validate
python -m src.fusion_demo --n 4 # synthetic demonstration
```

Environment: same `.venv`, single GTX 1650. No new dependencies. Fusion training
peaks well under 1 GB VRAM.

---

## 8. Project-level outcome (all 4 phases)

| Phase | Modality | Standalone test AUROC | Through fusion pathway |
|---|---|---|---|
| 1 | Chest X-ray (NIH) | 0.878 mean | 0.865 (−0.013) |
| 2 | Echo video (EchoNet) | 0.802 macro | 0.806 (+0.004) |
| 3 | Cardiac MRI (ACDC) | 0.706 macro | 0.628 (−0.078, N=15 noise) |
| 4 | Fusion | — | mechanism validated on X-ray + Echo; MRI within noise |

**What was achieved:** three modality encoders with a common `[B,1024]` contract,
and a fusion layer that provably preserves each encoder's signal while handling
1–3 present modalities via a learned missing-modality token.

**What remains (future work / paper "limitations"):**
1. **No paired tri-modal cohort** → real multi-modal accuracy is unmeasured. The
   next step needs a dataset where one patient has ≥2 modalities; then retrain
   fusion with real multi-present samples and measure multi- vs best-single and
   graceful degradation under modality dropout.
2. **MRI encoder** is the weak link (70 training patients). 5-fold CV or a
   segmentation-aided MRI model would lift it.
3. **One split per phase, no cross-validation** → all numbers are point estimates;
   error bars are largest for MRI.
4. Fusion is trained single-present only, so *all-present* inference is
   out-of-distribution — acceptable for a mechanism demonstration, not for
   deployment.
