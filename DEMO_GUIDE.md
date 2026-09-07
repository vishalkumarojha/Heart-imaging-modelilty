# Demo Guide — Multi-Modal Cardiac Condition Detection System

Quick reference for the live capstone presentation.

## Launch

```bash
cd Capstone
. .venv/bin/activate          # or: source .venv/bin/activate
python demo_app.py
```

Opens on **http://127.0.0.1:7860**. First prediction in each tab takes ~2–3 s
(loads that checkpoint once); every prediction after is < 1 s. **CPU only** — no
GPU needed, nothing to configure. Leave the terminal running; `Ctrl-C` to stop.

**Pre-flight (do this before the audience is watching):** open the app, click
through all four tabs once with the sample files so every checkpoint is warm.

Sample files are in `demo_samples/` — see `demo_samples/README.md` for each
file's true label and expected output.

---

## Tab 1 · Chest X-ray

**What it is:** DenseNet121 (ImageNet-pretrained), Phase 1. Multi-label —
Cardiomegaly and Effusion are scored **independently** (not a distribution).

**Demo flow:** upload `demo_samples/xray/both_cardiomegaly+effusion__*.png` →
both bars near 100 %. Then `no_finding__*.png` → both near 0 %. Then
`cardiomegaly_only__*.png` → one high, one low.

**Talking points:**
- Trained on NIH ChestX-ray14, 109k images, **patient-level** train/val/test
  split (no patient in two splits).
- **Test mean AUROC 0.878** (Cardiomegaly 0.897, Effusion 0.859) — and test
  *beat* validation, so it generalises.
- We report **AUROC, not accuracy**: Cardiomegaly is ~2.5 % of images, so a
  "always negative" model is 97 % accurate and useless.
- Low precision at the 0.5 threshold is deliberate — the loss up-weights the
  rare positive class; you'd tune the threshold per deployment.

---

## Tab 2 · Echocardiogram

**What it is:** ResNet18 applied per frame + a bidirectional LSTM over 16
evenly-sampled frames, Phase 2. Single-label 3-class ejection-fraction category.

**Demo flow:** upload `reduced_EF22__*.avi` → "Reduced" ~97 %. Then
`normal_EF71__*.avi` → "Normal" ~98 %. Then `mildly_reduced_EF50__*.avi` →
"Mildly Reduced" ~72 % (point out this is the hard class).

**Talking points:**
- EF buckets: **Reduced < 40 · Mildly Reduced 40–54 · Normal ≥ 55**.
- Uses EchoNet-Dynamic's **official** train/val/test split (one video per patient).
- **Test macro one-vs-rest AUROC 0.802.** Per class: Reduced 0.906, Normal
  0.822, **Mildly Reduced 0.678**.
- The middle class is hard *by construction* — a 15-point EF window, and
  echo-derived EF itself carries ~±5 % measurement noise, so borderline cases
  are genuinely ambiguous.

---

## Tab 3 · Cardiac MRI

**What it is:** ResNet18 per short-axis slice + BiLSTM over 10 slices, Phase 3.
Each slice is a 3-channel image `[ED frame, ES frame, ED−ES]` — the difference
channel encodes contraction. Single-label 5-class diagnosis.

**Demo flow:** upload **both** files from `demo_samples/mri/DCM__patient002/` →
"DCM" ~84 %. Then `RV__patient084/` → "RV" ~89 %. Then `NOR__patient068/` →
"NOR" ~77 %.

**Talking points — be candid here:**
- Classes: **DCM** dilated / **HCM** hypertrophic / **MINF** prior infarct /
  **NOR** normal / **RV** abnormal right ventricle.
- ACDC has **only 100 patients** total → our split is **70 / 15 / 15**.
- **Test macro AUROC 0.706, accuracy 6/15.** DCM, NOR, RV classify reasonably;
  **HCM and MINF are near chance** — the model confuses thick-walled HCM with
  dilated DCM, and regional-wall-motion MINF needs segmentation we didn't use.
- This is an **honest, expected limitation of dataset size**, documented in
  `PHASE3_MRI_SUMMARY.md`. The MRI encoder's real value is as a **component of
  the fusion model**, not a standalone diagnostic.

---

## Tab 4 · Fusion  (the centerpiece)

**What it is:** Phase 4. The three frozen encoders → three 1024-d embeddings →
a fusion layer → three task-specific heads. Handles **any subset** of modalities.

**Demo flow:**
1. **One modality** — upload just an X-ray. Panel shows ECHO and MRI as
   "🔀 learned missing-modality token". The X-ray head still fires.
2. **Two modalities** — add an ECHO video. Now two "✅ real upload", one token.
3. **All three** — add an MRI folder's two files. **A large red banner appears:**
   *"SYNTHETIC EXAMPLE — NOT ONE REAL PATIENT."* All three heads fire.

**Talking points:**
- **Missing-modality token:** an absent modality is replaced by a *trained*
  1024-d parameter vector — **not zeros**. The model learns a meaningful
  "this modality is unavailable" prior.
- **Combination:** per-modality LayerNorm (X-ray embeddings are ~15× larger in
  magnitude) → concatenate (→ 3072) → shared 2-layer MLP (→ 512) → three heads.
  We chose concat+MLP over attention: only 3 fixed slots, nothing to route.
- **We kept each dataset's real labels** (X-ray 2-label, EF 3-class, diagnosis
  5-class) rather than inventing a unified label — so we can measure exactly
  what the fusion pathway costs.
- **Validation — single-modality-present** (each real test set, other two
  masked): X-ray mean AUROC **0.865** (vs 0.878 standalone), ECHO macro AUROC
  **0.806** (vs 0.802), MRI **0.628** (vs 0.706, within noise at N=15). **The
  fusion pathway preserves each encoder's signal** — that's the result.
- **The honest caveat, say it plainly:** the three datasets come from three
  institutions with **no shared patients**, so there is **no real tri-modal
  patient** anywhere. We cannot and do not report a real multi-modal accuracy
  number. The all-three-present mode is an **architecture demonstration** on
  three unrelated people's scans — hence the banner. Measuring true fusion
  performance needs a paired cohort, which is the stated future work.

---

## If something goes wrong live

- **Bad upload / wrong file type** → the tab shows a plain `⚠️` message, not a
  crash. Just pick a `demo_samples/` file and retry.
- **MRI tab error "Need 2 volume files"** → you uploaded one file, or included a
  `_gt` mask. Upload exactly the two `_frameNN.nii.gz` files from one folder.
- **Port 7860 in use** → `python demo_app.py` after `pkill -f demo_app` (or edit
  `server_port` at the bottom of `demo_app.py`).
- **App won't start** → check the three encoder checkpoints exist under
  `outputs/checkpoints/{,echo/,mri/,fusion/}`; they are git-ignored (large), so
  a fresh clone needs them copied in or regenerated.
