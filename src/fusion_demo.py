"""Phase 4 — SYNTHETIC multi-modal demonstration.

╔══════════════════════════════════════════════════════════════════════════╗
║  SYNTHETIC DEMONSTRATION — NOT A REAL PATIENT.                           ║
║                                                                          ║
║  NIH ChestX-ray14, EchoNet-Dynamic and ACDC share NO patients. This     ║
║  script fabricates "patients" by pairing an X-ray embedding from one     ║
║  real person, an echo embedding from a DIFFERENT real person, and an     ║
║  MRI embedding from a THIRD real person, then runs them through the      ║
║  fusion model with all 3 modalities marked present.                     ║
║                                                                          ║
║  The outputs below are an ILLUSTRATION of the fused forward pass only.   ║
║  They are NOT validated multi-modal performance and must never be        ║
║  reported as such. The only real fusion numbers are in                  ║
║  single_modality_validation.csv (see fusion_train.py).                  ║
╚══════════════════════════════════════════════════════════════════════════╝

    python -m src.fusion_demo [--n 5] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from . import fusion_config as FC
from .fusion_model import FusionModel
from .utils import setup_logging

logger = setup_logging()

BANNER = "SYNTHETIC DEMONSTRATION — not a real patient"

# heuristic-only mapping of each head's output to a 0/1/2 "cardiac concern" level.
# NOT trained, NOT validated — an illustrative post-hoc combination for the demo.
_ECHO_RISK = {0: 2, 1: 1, 2: 0}                       # Reduced/Mildly/Normal
_MRI_RISK = {"DCM": 2, "MINF": 2, "HCM": 1, "RV": 1, "NOR": 0}
_RISK_NAME = {0: "LOW", 1: "MODERATE", 2: "HIGH"}


def _load_cache(mod: str, split: str = "test"):
    p = FC.EMB_CACHE_DIR / f"{mod}_{split}.npz"
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python -m src.fusion_train` first (it precomputes embeddings).")
    d = np.load(p, allow_pickle=True)
    return d["emb"].astype(np.float32), d["label"], list(d["id"])


def _heuristic_risk(xray_p, echo_pred, mri_pred_name) -> str:
    xr = int(xray_p[0] >= 0.5) + int(xray_p[1] >= 0.5)          # 0/1/2 findings
    er = _ECHO_RISK[int(echo_pred)]
    mr = _MRI_RISK[mri_pred_name]
    return _RISK_NAME[max(xr, er, mr)]


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="SYNTHETIC multi-modal fusion demo")
    ap.add_argument("--n", type=int, default=5, help="number of synthetic 'patients'")
    ap.add_argument("--seed", type=int, default=FC.SEED)
    ap.add_argument("--checkpoint", default=str(FC.CKPT_BEST))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    logger.info("#" * 74)
    logger.info("#  %s", BANNER.upper())
    logger.info("#  3 unrelated real people (one per dataset) fused into one fake 'patient'.")
    logger.info("#  Illustrates the fused forward pass. NOT multi-modal performance.")
    logger.info("#" * 74)

    model = FusionModel().to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    logger.info("fusion checkpoint: %s (epoch %s, scheme=%s)",
                FC.CKPT_BEST.name, ck.get("epoch"), ck.get("scheme"))

    xe, xl, xid = _load_cache("xray")
    ee, el, eid = _load_cache("echo")
    me, ml, mid = _load_cache("mri")
    echo_cls = FC.TASK_SPECS["echo"]["classes"]
    mri_cls = FC.TASK_SPECS["mri"]["classes"]
    xray_cls = FC.TASK_SPECS["xray"]["classes"]

    rows = []
    for k in range(args.n):
        xi, ei, mi = int(rng.integers(len(xe))), int(rng.integers(len(ee))), int(rng.integers(len(me)))
        embeds = [torch.from_numpy(xe[xi:xi + 1]).to(device),
                  torch.from_numpy(ee[ei:ei + 1]).to(device),
                  torch.from_numpy(me[mi:mi + 1]).to(device)]
        with torch.no_grad():
            out_all, _ = model(embeds, model.mask_for(FC.MODALITIES, 1, device))     # all present
        xray_p = torch.sigmoid(out_all["xray"])[0].cpu().numpy()
        echo_p = torch.softmax(out_all["echo"].float(), 1)[0].cpu().numpy()
        mri_p = torch.softmax(out_all["mri"].float(), 1)[0].cpu().numpy()
        echo_pred, mri_pred = int(echo_p.argmax()), int(mri_p.argmax())
        risk = _heuristic_risk(xray_p, echo_pred, mri_cls[mri_pred])

        logger.info("-" * 74)
        logger.info("[%s]  SYNTHETIC patient #%d", BANNER, k + 1)
        logger.info("  composed from  X-ray=%s (NIH)  +  echo=%s (EchoNet)  +  MRI=%s (ACDC)  "
                    "— three different real people",
                    xid[xi], eid[ei], mid[mi])
        logger.info("  X-ray head : P(Cardiomegaly)=%.2f  P(Effusion)=%.2f", xray_p[0], xray_p[1])
        logger.info("  echo  head : %s  (%s)",
                    echo_cls[echo_pred], " ".join(f"{c}={p:.2f}" for c, p in zip(echo_cls, echo_p)))
        logger.info("  MRI   head : %s  (%s)",
                    mri_cls[mri_pred], " ".join(f"{c}={p:.2f}" for c, p in zip(mri_cls, mri_p)))
        logger.info("  >> heuristic fused cardiac-concern level: %s   "
                    "(NOT a trained/validated output — worst-of-3 illustrative rule)", risk)

        rows.append({
            "SYNTHETIC": "TRUE  (not a real patient)",
            "synthetic_id": k + 1,
            "xray_source_id_NIH": xid[xi], "echo_source_id_EchoNet": eid[ei], "mri_source_id_ACDC": mid[mi],
            "xray_P_Cardiomegaly": round(float(xray_p[0]), 4),
            "xray_P_Effusion": round(float(xray_p[1]), 4),
            "echo_pred": echo_cls[echo_pred],
            "mri_pred": mri_cls[mri_pred],
            "heuristic_risk_level": risk,
        })

    with FC.SYNTHETIC_DEMO_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    logger.info("=" * 74)
    logger.info("wrote %s", FC.SYNTHETIC_DEMO_CSV.name)
    logger.info("REMINDER: %s. Every row above pairs THREE UNRELATED real people.", BANNER)
    logger.info("Real, validated numbers live ONLY in single_modality_validation.csv.")


if __name__ == "__main__":
    main()
