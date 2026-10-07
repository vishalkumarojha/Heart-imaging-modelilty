"""Phase 7b — Grad-CAM case study on the frozen model.

Deterministic: the case list comes from `outputs/gradcam/case_list.json`
(written by `src.error_analysis.py`, selected without any RNG), the checkpoint
is the frozen baseline, and the transform is the evaluation transform. Running
it twice on two machines produces the same overlays for the same cases.

Outputs
    outputs/gradcam/overlays/<case_id>__<label>.png   image + heatmap overlay
    outputs/gradcam/gradcam_summary.json              per-case CAM statistics
    outputs/gradcam/bbox_localization.json            where a radiologist box
                                                      exists for the case

Quantitative honesty: Grad-CAM is *qualitative*. Where NIH's
`BBox_List_2017.csv` has a box for the case's label we additionally report
    * pointing-game hit  = argmax(heat) inside the box,
    * heat mass inside the box vs the box's share of the image area
      (ratio > 1 means the model looks at the box more than chance),
and we say nothing at all for cases without a box.

    python -m src.gradcam [--limit N] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .inference import load_model
from .reproducibility import capture_environment, sha256_file, write_json
from .utils import setup_logging

logger = setup_logging()

# last conv block of DenseNet121 inside our encoder
TARGET_LAYER = "encoder.features.denseblock4"


# --------------------------------------------------------------------------- #
# Grad-CAM core
# --------------------------------------------------------------------------- #
class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) for `MultiLabelClassifier`.

    Only one forward+backward pair is needed per case: gradients of the target
    class score are averaged spatially to weight the feature channels.
    """

    def __init__(self, model, layer_path: str = TARGET_LAYER):
        import torch

        self.model = model
        self.model.eval()
        self._acts: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

        layer = dict(model.named_modules())[layer_path]
        if layer is None or not hasattr(layer, "register_forward_hook"):
            raise ValueError(f"layer {layer_path!r} not found in model")
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):  # noqa: N802
        import torch

        self._acts = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):  # noqa: N802
        self._grads = grad_out[0].detach()

    def __call__(self, x, class_idx: int) -> np.ndarray:
        import torch

        self.model.zero_grad(set_to_none=True)
        score = self.model(x)[:, class_idx].sum()
        score.backward()
        if self._acts is None or self._grads is None:
            raise RuntimeError("hooks did not fire — check TARGET_LAYER")
        weights = self._grads.mean(dim=(2, 3), keepdim=True)   # GAP of grads
        cam = torch.relu((weights * self._acts).sum(dim=1))    # (B,H,W)
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1), size=x.shape[-2:], mode="bilinear",
            align_corners=False).squeeze(1)
        out = cam[0].cpu().numpy()
        lo, hi = float(out.min()), float(out.max())
        if hi - lo > 1e-12:
            out = (out - lo) / (hi - lo)
        return out


# --------------------------------------------------------------------------- #
# Bounding-box reference (only where NIH provides one)
# --------------------------------------------------------------------------- #
def load_bbox_reference() -> Dict[Tuple[str, str], Tuple[float, float, float, float]]:
    """{(image_index, label): (x, y, w, h)} in original-image pixels.

    The NIH header is `Image Index,Finding Label,Bbox [x,y,w,h],,,`, so the
    coordinate columns arrive named `Bbox [x`, `y`, `w`, `h]` — we take the
    first six fields positionally instead of trusting the names.
    """
    box_path = C.DATA_RAW_DIR / "BBox_List_2017.csv"
    if not box_path.exists():
        return {}
    df = pd.read_csv(box_path, header=0, usecols=range(6))
    df.columns = ["Image Index", "Finding Label", "x", "y", "w", "h"]
    df = df.dropna(subset=["x", "y", "w", "h"])
    return {(str(r["Image Index"]), str(r["Finding Label"])):
            (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]))
            for _, r in df.iterrows()}


def bbox_localization(cam: np.ndarray, box: Tuple[float, float, float, float],
                      image_size: Tuple[int, int]) -> Dict[str, float]:
    """Score one heat map against one reference box (image_size = (W, H))."""
    w_img, h_img = image_size
    x, y, bw, bh = box
    h_map, w_map = cam.shape
    sx, sy = w_map / w_img, h_map / h_img
    x0, x1 = max(int(np.floor(x * sx)), 0), min(int(np.ceil((x + bw) * sx)), w_map)
    y0, y1 = max(int(np.floor(y * sy)), 0), min(int(np.ceil((y + bh) * sy)), h_map)
    inside = cam[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else np.zeros((0, 0))
    total = float(cam.sum()) + 1e-12
    mass_inside = float(inside.sum())
    area_frac = ((x1 - x0) * (y1 - y0)) / float(w_map * h_map)
    argmax = np.unravel_index(int(cam.argmax()), cam.shape)
    hit = bool(x0 <= argmax[1] < x1 and y0 <= argmax[0] < y1)
    return {
        "pointing_game_hit": hit,
        "heat_mass_inside": mass_inside / total,
        "box_area_fraction": area_frac,
        "concentration_ratio": (mass_inside / total) / max(area_frac, 1e-9),
    }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def bbox_cohort(model, cam_engine, label_names, transform, device,
                split: str = "test") -> List[Dict[str, object]]:
    """Grad-CAM localization over every test-split image that HAS a NIH box.

    The hand-picked case list rarely overlaps the 984 boxed images, so the
    quantitative localization number comes from this full boxed subset instead
    — a fixed, cohort-wide set, not a cherry-picked one.
    """
    from PIL import Image

    from .dataset import scan_image_paths

    box_path = C.DATA_RAW_DIR / "BBox_List_2017.csv"
    if not box_path.exists():
        return []
    split_path = C.DATA_PROCESSED_DIR / "split_index.csv"
    if not split_path.exists():
        return []
    split_df = pd.read_csv(split_path)
    test_ids = set(split_df.loc[split_df["split"] == split, "Image Index"])
    path_map = scan_image_paths()

    pairs = load_bbox_reference()
    rows: List[Dict[str, object]] = []
    for (img, label), box in sorted(pairs.items()):
        if img not in test_ids or label not in label_names:
            continue
        path = path_map.get(img)
        if not path:
            continue
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            native = rgb.size
            tensor = transform(image=np.array(rgb))["image"].unsqueeze(0).to(device)
        heat = cam_engine(tensor, label_names.index(label))
        rows.append({"image_index": img, "label": label, "box": list(box),
                     **bbox_localization(heat, box, native)})
    return rows


def run(limit: Optional[int] = None, device: Optional[str] = None) -> Dict[str, object]:
    import torch
    from PIL import Image

    from .dataset import build_transforms

    case_file = C.GRADCAM_DIR / "case_list.json"
    if not case_file.exists():
        raise FileNotFoundError(
            f"{case_file} missing — run `python -m src.error_analysis` first")
    payload = json.loads(case_file.read_text())
    cases: List[dict] = payload["cases"]
    if limit:
        cases = cases[: int(limit)]

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, label_names = load_model(C.BASELINE_CHECKPOINT_BEST, dev)
    cam_engine = GradCAM(model)
    transform = build_transforms(train=False)

    overlay_dir = C.GRADCAM_DIR / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    per_case: List[Dict[str, object]] = []
    for case in cases:
        path = Path(case["path"])
        label = case["label"]
        class_idx = label_names.index(label)
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            tensor = transform(image=np.array(rgb))["image"].unsqueeze(0).to(dev)
        with torch.no_grad():
            prob = float(torch.sigmoid(model(tensor)[0, class_idx]))
        heat = cam_engine(tensor, class_idx)

        cam_uint8 = (heat * 255).astype(np.uint8)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        base = np.array(rgb.resize((heat.shape[1], heat.shape[0]))).astype(float)
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
        axes[0].imshow(base.astype(np.uint8), cmap="gray")
        axes[0].set_title("input", fontsize=9)
        axes[1].imshow(heat, cmap="jet")
        axes[1].set_title(f"Grad-CAM — {label}", fontsize=9)
        axes[2].imshow(base.astype(np.uint8), cmap="gray")
        axes[2].imshow(heat, cmap="jet", alpha=0.45)
        axes[2].set_title(
            f"{case['stratum']}  p={prob:.3f}  τ={case['threshold']:.3f}", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        out_png = overlay_dir / f"{case['case_id']}__{label.lower()}.png"
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)

        entry = {**{k: case[k] for k in
                    ("case_id", "label", "stratum", "image_index", "policy",
                     "threshold", "probability", "certainty")},
                 "checkpoint_probability": round(prob, 4),
                 "overlay": str(out_png.relative_to(C.PROJECT_ROOT)),
                 "cam_entropy": round(float(-(heat[heat > 0] *
                                               np.log(heat[heat > 0] + 1e-12)).sum()
                                             / max((heat > 0).sum(), 1)), 4)}
        per_case.append(entry)

        logger.info("CAM %s -> %s", case["case_id"], out_png.name)

    summary = {
        "phase": "gradcam",
        "status": "COMPLETED",
        "checkpoint": str(C.BASELINE_CHECKPOINT_BEST),
        "checkpoint_sha256": sha256_file(C.BASELINE_CHECKPOINT_BEST),
        "target_layer": TARGET_LAYER,
        "n_cases": len(per_case),
        "cases": per_case,
        "case_list_source": str(case_file.relative_to(C.PROJECT_ROOT)),
        "quantitative_note": (
            "Grad-CAM is qualitative. A numeric localization score is reported "
            "only for the subset of cases that have a radiologist bounding box "
            "in NIH's BBox_List_2017.csv; cases without a box get no score."),
        "environment": capture_environment(include_packages=False),
    }
    summary_path = write_json(C.GRADCAM_DIR / "gradcam_summary.json", summary)

    bbox_hits = bbox_cohort(model, cam_engine, label_names, transform, dev, payload["split"])
    if bbox_hits:
        hits = [b["pointing_game_hit"] for b in bbox_hits]
        ratios = [b["concentration_ratio"] for b in bbox_hits]
        write_json(C.GRADCAM_DIR / "bbox_localization.json", {
            "cohort": f"{payload['split']} images with a NIH radiologist box, "
                      "for the boxed label only (fixed set, no cherry-picking)",
            "n_with_box": len(bbox_hits),
            "n_labels": len({b["label"] for b in bbox_hits}),
            "pointing_game_hits": int(sum(hits)),
            "pointing_game_accuracy": float(np.mean(hits)),
            "mean_concentration_ratio": float(np.mean(ratios)),
            "ratio_above_1": int(sum(r > 1.0 for r in ratios)),
            "median_concentration_ratio": float(np.median(ratios)),
            "cases": bbox_hits,
        })
        logger.info("bbox cohort: n=%d pointing=%.2f mean ratio=%.2f",
                    len(bbox_hits), float(np.mean(hits)), float(np.mean(ratios)))
    else:
        logger.warning(
            "No boxed test image found — Grad-CAM stays qualitative "
            "(no localization number will be reported).")

    logger.info("Grad-CAM -> %s (%d cases)", summary_path, len(per_case))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7b: Grad-CAM case study")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.limit, args.device)
