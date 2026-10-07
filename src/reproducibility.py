"""Reproducibility helpers: environment capture, hashing, JSON artifacts.

Used by the baseline freeze (Phase 2) and every research experiment so that any
results file can be traced back to the exact code, data and hardware that
produced it.  Nothing here imports torch at module level — capture is lazy so
`sha256_file` / `write_json` stay cheap for unit tests.

    from src.reproducibility import capture_environment, sha256_file, write_json
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (checkpoints are ~84 MB)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(digest: str, n: int = 12) -> str:
    return digest[:n]


# --------------------------------------------------------------------------- #
# JSON artifacts
# --------------------------------------------------------------------------- #
def _default(obj: Any) -> Any:
    """JSON encoder for numpy / pathlib scalars that may appear in results."""
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: Dict[str, Any], indent: int = 2) -> Path:
    """Atomic JSON write (tmp + replace) so a crash never truncates a result file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent, default=_default))
    tmp.replace(path)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Environment capture
# --------------------------------------------------------------------------- #
def capture_environment(include_packages: bool = True) -> Dict[str, Any]:
    """Detect (never invent) the runtime environment.

    Returns a dict with python / torch / torchvision / CUDA / GPU / CPU / OS and
    the versions of the packages the project actually imports.
    """
    env: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or _cpu_model(),
    }

    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_runtime"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "capability": f"{props.major}.{props.minor}",
            }
        else:
            env["gpu"] = None
    except ImportError:
        env["torch"] = None

    try:
        import torchvision

        env["torchvision"] = torchvision.__version__
    except ImportError:
        env["torchvision"] = None

    if include_packages:
        env["packages"] = _package_versions()
    return env


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:  # pragma: no cover - non-Linux
        pass
    return ""


_WANTED_PACKAGES = (
    "numpy", "pandas", "scikit-learn", "albumentations", "Pillow",
    "tqdm", "matplotlib", "opencv-python-headless", "nibabel", "gradio", "pytest",
)


def _package_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        return out
    for name in _WANTED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def capture_split_statistics(frame) -> Dict[str, Any]:
    """Row/patient/label counts per split for the frozen X-ray split frame."""
    labels = [c for c in ("Cardiomegaly", "Effusion") if c in frame.columns]
    stats: Dict[str, Any] = {
        "rows": int(len(frame)),
        "patients": int(frame["Patient ID"].nunique()),
        "rows_by_split": {k: int(v) for k, v in frame["split"].value_counts().items()},
        "patients_by_split": {
            k: int(v) for k, v in frame.groupby("split")["Patient ID"].nunique().items()
        },
        "label_counts_by_split": {
            lbl: {k: int(v) for k, v in frame.groupby("split")[lbl].sum().items()}
            for lbl in labels
        },
        "label_counts_total": {lbl: int(frame[lbl].sum()) for lbl in labels},
        "patient_overlap": _patient_overlap(frame),
    }
    return stats


def _patient_overlap(frame) -> Dict[str, int]:
    """Explicit leakage check: pairwise patient intersection sizes (must be 0)."""
    per_split = {s: set(frame.loc[frame["split"] == s, "Patient ID"]) for s in ("train", "val", "test")}
    return {
        "train_val": len(per_split["train"] & per_split["val"]),
        "train_test": len(per_split["train"] & per_split["test"]),
        "val_test": len(per_split["val"] & per_split["test"]),
    }
