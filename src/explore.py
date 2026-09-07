"""Data exploration / sanity report. Run BEFORE any training.

    python -m src.explore

Prints:
  * total image files found on disk (per folder + total)
  * how many Data_Entry_2017.csv rows have a matching file (and flags mismatches)
  * whether official NIH split files were found, or a patient-level split was built
  * label distribution (Cardiomegaly / Effusion) overall and per split
  * device / CUDA / VRAM report
"""
from __future__ import annotations

import json

from . import config as C
from .dataset import build_dataset_frame, summarize
from .utils import setup_logging

logger = setup_logging()


def device_report() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        logger.warning("torch not importable yet - skipping device report.")
        return

    logger.info("torch %s | torchvision available: %s", torch.__version__, _torchvision_version())
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        logger.info(
            "CUDA OK: %s | %.2f GB VRAM | capability %d.%d | CUDA runtime %s",
            torch.cuda.get_device_name(idx),
            props.total_memory / (1024 ** 3),
            props.major, props.minor,
            torch.version.cuda,
        )
    else:
        logger.warning("CUDA NOT available - CPU only.")


def _torchvision_version() -> str:
    try:
        import torchvision

        return torchvision.__version__
    except Exception as e:  # pragma: no cover
        return f"<not importable: {e}>"


def main() -> None:
    logger.info("=" * 70)
    logger.info("DATA EXPLORATION")
    logger.info("=" * 70)

    frame, report = build_dataset_frame(use_cache=False, write_cache=True)

    logger.info("-" * 70)
    if report.get("from_cache"):
        logger.info("(loaded from cache)")
    else:
        logger.info("Images on disk (unique files) : %d", report["images_on_disk"])
        logger.info("Data_Entry_2017.csv rows      : %d", report["csv_entries"])
        logger.info("CSV rows WITH a file on disk  : %d", report["csv_entries"] - report["csv_without_file"])
        logger.info("CSV rows WITHOUT a file       : %d  %s",
                    report["csv_without_file"],
                    "<-- MISMATCH" if report["csv_without_file"] else "(clean)")
        logger.info("Files on disk NOT in CSV       : %d  %s",
                    report["disk_without_csv"],
                    "<-- MISMATCH" if report["disk_without_csv"] else "(clean)")
        logger.info("Usable rows (used downstream)  : %d", report["usable_rows"])
        logger.info("-" * 70)
        logger.info("Split method: %s", report["split_method"])
        logger.info("Split row counts: %s", report["split_counts"])
        logger.info("Split patient counts: %s", _patient_counts(frame))

    logger.info("-" * 70)
    summarize(frame, C.TARGET_LABELS)

    logger.info("-" * 70)
    device_report()

    logger.info("-" * 70)
    logger.info("Cache files written:")
    logger.info("  %s", C.IMAGE_PATH_CACHE)
    logger.info("  %s", C.SPLIT_CACHE_CSV)
    if not report.get("from_cache"):
        (C.LOG_DIR / "exploration_report.json").write_text(json.dumps(report, indent=2, default=str))
        logger.info("  %s", C.LOG_DIR / "exploration_report.json")


def _patient_counts(frame) -> dict:
    return frame.groupby("split")["Patient ID"].nunique().to_dict()


if __name__ == "__main__":
    main()
