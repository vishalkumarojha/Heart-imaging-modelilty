# Capstone — Phase 1: Cardiomegaly & Effusion from Chest X-rays

Multi-label DenseNet121 classifier on the NIH ChestX-ray14 dataset. This is
**Phase 1** of a multi-modality heart-condition detection system (X-ray now;
ECHO and MRI later). The code is intentionally split so the X-ray backbone
(`src/model.py::XrayEncoder`) can be dropped into a shared fusion layer without
dragging X-ray-specific data logic along with it.

## Layout

```
Capstone/
├── data/
│   ├── raw/            NIH ChestX-ray14 (Data_Entry_2017.csv, images_001..012/, ...)
│   └── processed/      cached image-path index + split assignment (auto-generated)
├── src/
│   ├── config.py       all paths / hyper-params / RunConfig
│   ├── utils.py        seeding, logging, AUROC + sensitivity/specificity helpers
│   ├── dataset.py      folder scan, label parsing, patient-level split, transforms, loaders
│   ├── model.py        DenseNet121 encoder + head, freeze_backbone() / unfreeze_all()
│   ├── explore.py      data sanity report (run this first)
│   ├── train.py        two-phase training loop
│   └── evaluate.py     test-set metrics + predictions CSV
├── outputs/
│   ├── checkpoints/    densenet121_best.pt (best mean val AUROC)
│   └── logs/           metrics.csv, test_predictions.csv, exploration_report.json
├── requirements.txt
└── README.md
```

## Setup

```bash
cd Capstone
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

All commands below are run from `Capstone/` as modules (`python -m src.<name>`)
so the package imports resolve.

## 1. Data exploration (run first)

```bash
python -m src.explore
```

Reports: images found on disk, CSV-vs-disk match, which split strategy was used
(official NIH files if `train_val_list.txt` + `test_list.txt` are present under
`data/raw/`, otherwise a **patient-level** 70/15/15 split — no patient leaks
across splits), label distribution for Cardiomegaly / Effusion, and the
CUDA/GPU/VRAM report. Writes `data/processed/split_index.csv` (the cached master
table used by training and eval).

## 2. Quick smoke test

```bash
python -m src.train --quick_test
```

500 train / 100 val images, 1 epoch, Phase 1 only. Confirms the whole pipeline
(data → model → loss → checkpoint → metrics.csv) runs end-to-end.

## 3. Full training

```bash
python -m src.train
```

- **Phase 1** (5 epochs): backbone frozen except the final dense block + the
  classifier, `lr = 1e-4`.
- **Phase 2** (5 epochs): fully unfrozen, `lr = 1e-5`.
- `BCEWithLogitsLoss` (with per-label `pos_weight` from train prevalence to
  handle the heavy negative imbalance), Adam optimizer.
- Mixed precision on CUDA by default (`--no_amp` to disable).
- **`densenet121_best.pt`** — highest **mean val AUROC** across both labels,
  saved on any epoch of either phase it improves.
- **`densenet121_last.pt`** — written every epoch. Resume an interrupted run
  with `python -m src.train --resume` (restores model + optimizer + AMP scaler
  + epoch counter; completed epochs are skipped, `metrics.csv` is appended to).
- Per-epoch `train_loss` / `val_loss` / per-label + mean val AUROC appended to
  `outputs/logs/metrics.csv`.

Useful flags: `--resume`, `--batch_size N`, `--num_workers N`, `--no_amp`.

Long runs: launch detached and tail the log —
```bash
nohup .venv/bin/python -m src.train > outputs/logs/train_run.log 2>&1 &
echo $! > outputs/logs/train.pid
tail -f outputs/logs/train_run.log
```

## 4. Evaluation

```bash
python -m src.evaluate
```

Loads the best checkpoint, runs the **test** split, prints a per-label table of
AUROC / sensitivity / specificity / precision / F1 (threshold 0.5), and writes
`outputs/logs/test_predictions.csv` (image index, patient id, ground truth,
score, prediction per label).

## Design notes

- **AUROC is the primary metric**, not accuracy — prevalence of both target
  conditions is low, so accuracy is misleading.
- **No horizontal flip** augmentation: chest X-ray laterality is diagnostic.
  Training augmentation is limited to ±10° rotation and mild
  brightness/contrast jitter.
- **Patient-level split**: images from one `Patient ID` never span
  train/val/test.
- **Modularity for fusion**: `XrayEncoder` outputs a 1024-d feature vector and
  knows nothing about the split, the CSV, or the number of target labels.
  `MultiLabelImageDataset` is modality-agnostic (`path` column + label columns).
  Later, a fusion model concatenates `XrayEncoder` / `EchoEncoder` /
  `MriEncoder` features into a shared head.
