# Demo sample files

Curated known-good files from the **held-out test sets** of each phase, chosen
for clear, unambiguous ground truth so the live demo shows sensible results.
Expected model output is what the trained checkpoints produced during evaluation.

## X-ray (`xray/`) — upload one file in Tab 1 (or Tab 4)

| File | True label | Expect the model to show |
|---|---|---|
| `both_cardiomegaly+effusion__00004533_023.png` | Cardiomegaly **+** Effusion | Cardiomegaly ≈ 100 %, Effusion ≈ 96 % (both above 0.5) |
| `cardiomegaly_only__00003817_000.png` | Cardiomegaly only | Cardiomegaly ≈ 95 %, Effusion ≈ 17 % (only Cardiomegaly above 0.5) |
| `no_finding__00000047_000.png` | neither | Cardiomegaly ≈ 0 %, Effusion ≈ 2 % (both below 0.5) |

## ECHO (`echo/`) — upload one `.avi` in Tab 2 (or Tab 4)

| File | True EF category (EF %) | Expect |
|---|---|---|
| `reduced_EF22__0X28712788DD9BC1B6.avi` | **Reduced** (EF ≈ 22) | "Reduced" ≈ 97 % |
| `mildly_reduced_EF50__0X30C6E212959E3A90.avi` | **Mildly Reduced** (EF ≈ 50) | "Mildly Reduced" ≈ 72 % (this class is intrinsically the hardest) |
| `normal_EF71__0X5ECD55B830580E47.avi` | **Normal** (EF ≈ 71) | "Normal" ≈ 98 % |

## MRI (`mri/`) — upload **both** `.nii.gz` files from one sub-folder in Tab 3 (or Tab 4)

Each folder holds one patient's ED and ES frame volumes (`_frame01` = ED,
`_frame12`/`_frame10` = ES). Upload **both files together**; do **not** upload the
`*_4d.nii.gz` or `*_gt.nii.gz` files (they aren't in these folders).

| Folder | True diagnosis | Expect |
|---|---|---|
| `DCM__patient002/` | **DCM** — dilated cardiomyopathy | "DCM" ≈ 84 % |
| `NOR__patient068/` | **NOR** — normal | "NOR" ≈ 77 % |
| `RV__patient084/` | **RV** — abnormal right ventricle | "RV" ≈ 89 % |

> MRI predictions are the least reliable (trained on 70 patients, test N = 15,
> macro AUROC 0.706) — these three are cases the model gets right; HCM and MINF
> are near chance and are deliberately not included as "known-good" samples.

## Tab 4 (Fusion) notes

- Upload any **1 or 2** of the above → the other modalities use the learned
  missing-modality token; the app labels each as "real upload" vs "token".
- Upload **all 3** (e.g. one X-ray + one ECHO + one MRI folder) → a red
  **"SYNTHETIC EXAMPLE — NOT ONE REAL PATIENT"** banner appears, because these
  three files are from three unrelated people in three different datasets.
