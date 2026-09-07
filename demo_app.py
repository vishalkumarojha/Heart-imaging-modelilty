"""Multi-Modal Cardiac Condition Detection System — Capstone Demo (Gradio app).

    python demo_app.py            # -> http://127.0.0.1:7860

4 tabs: X-ray · ECHO · MRI · Fusion. Upload a file, get a live prediction from
the frozen Phase 1-3 encoders + their trained heads, and (tab 4) the Phase 4
fusion pathway. All inference runs on CPU (encoders are small & frozen — a
single prediction is ~1-3 s).

Real / synthetic transparency: the Fusion tab shows a prominent red banner
whenever all three modalities are supplied together, because no real patient in
any of the three source datasets has more than one modality.
"""
from __future__ import annotations

import os
import re
import traceback

import gradio as gr

TITLE = "Multi-Modal Cardiac Condition Detection System — Capstone Demo"

XRAY_CLASSES = ["Cardiomegaly", "Effusion"]
ECHO_CLASSES = ["Reduced", "Mildly Reduced", "Normal"]
MRI_CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]
MRI_LONG = {
    "DCM": "dilated cardiomyopathy", "HCM": "hypertrophic cardiomyopathy",
    "MINF": "prior myocardial infarction", "NOR": "normal", "RV": "abnormal right ventricle",
}
CKPT = {
    "xray": "outputs/checkpoints/densenet121_best.pt",
    "echo": "outputs/checkpoints/echo/echo_cnn_lstm_best.pt",
    "mri": "outputs/checkpoints/mri/mri_resnet18_bilstm_best.pt",
    "fusion": "outputs/checkpoints/fusion/fusion_best.pt",
}

BIG_CSS = """
.gradio-container {max-width: 1150px !important; margin: auto;}
#hdr h1 {font-size: 1.7rem; margin-bottom: 0;}
.bigtext, .bigtext p {font-size: 1.03rem; line-height: 1.6;}
.synthetic-banner {
    background:#b3261e; color:#fff; font-size:1.15rem; font-weight:700;
    padding:16px 18px; border-radius:10px; border:3px solid #7a1a13; margin:6px 0;
}
"""

# --------------------------------------------------------------------------- #
# Lazy model cache  (loaded on first use so the app starts instantly)
# --------------------------------------------------------------------------- #
_CACHE: dict = {}


def _device():
    import torch

    return torch.device("cpu")           # deterministic + no CUDA surprises for a live demo


def _load(key: str):
    if key in _CACHE:
        return _CACHE[key]
    import torch

    dev = _device()
    if key == "xray":
        from src.model import build_model
        m = build_model(num_classes=2, pretrained=False)
        m.load_state_dict(torch.load(CKPT["xray"], map_location=dev)["model_state"])
    elif key == "echo":
        from src.echo_model import build_echo_model
        m = build_echo_model(num_classes=3, pretrained=False)
        m.load_state_dict(torch.load(CKPT["echo"], map_location=dev)["model_state"])
    elif key == "mri":
        from src.mri_model import build_mri_model
        m = build_mri_model(num_classes=5, pretrained=False)
        m.load_state_dict(torch.load(CKPT["mri"], map_location=dev)["model_state"])
    elif key == "fusion":
        from src.fusion_model import FusionModel
        from src.load_encoders import load_all_encoders
        bundles = load_all_encoders(dev, freeze=True)
        fm = FusionModel().to(dev)
        fm.load_state_dict(torch.load(CKPT["fusion"], map_location=dev)["model_state"])
        fm.eval()
        _CACHE[key] = (bundles, fm)
        return _CACHE[key]
    else:
        raise KeyError(key)
    m.to(dev).eval()
    _CACHE[key] = m
    return m


# --------------------------------------------------------------------------- #
# Preprocessing  (reuses the exact Phase 1-3 pipelines)
# --------------------------------------------------------------------------- #
def _prep_xray(path):
    import numpy as np
    from PIL import Image

    from src.dataset import build_transforms

    img = np.array(Image.open(path).convert("RGB"))
    t = build_transforms(train=False, image_size=224)(image=img)["image"]
    return t.unsqueeze(0)                                   # [1,3,224,224]


def _prep_echo(path):
    from src.echo_dataset import apply_clip, build_transforms, load_norm_stats, read_clip

    frames = read_clip(str(path), 16)                       # [16,H,W,3] uint8  (raises IOError if not a video)
    mean, std = load_norm_stats()
    clip = apply_clip(build_transforms(False, 112, mean, std), frames)
    return clip.unsqueeze(0)                                # [1,16,3,112,112]


def _resolve_ed_es(files):
    """From an uploaded file list, return (ed_path, es_path). ED = lowest frame #."""
    paths = [str(getattr(f, "name", f)) for f in files]
    vols = [p for p in paths if p.endswith((".nii.gz", ".nii")) and "_gt" not in os.path.basename(p)]
    if len(vols) < 2:
        raise ValueError(
            f"Need 2 volume files (the ED and ES frames, .nii.gz). Got {len(vols)} usable "
            f"of {len(paths)} uploaded. Do not upload the *_gt.nii.gz masks."
        )

    def frame_no(p):
        m = re.search(r"frame(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 10**9

    vols = sorted(vols, key=frame_no)
    return vols[0], vols[-1]


def _prep_mri(files):
    import numpy as np
    import torch

    from src.mri_config import MriRunConfig
    from src.mri_dataset import load_norm_stats, preprocess_pair

    ed, es = _resolve_ed_es(files)
    x = preprocess_pair(ed, es, MriRunConfig(), train=False, norm=load_norm_stats())
    return torch.from_numpy(np.asarray(x)).unsqueeze(0), os.path.basename(ed), os.path.basename(es)


# --------------------------------------------------------------------------- #
# Predict  (each wrapped by the tab handlers with try/except)
# --------------------------------------------------------------------------- #
def _predict_xray(path):
    import torch

    m = _load("xray")
    with torch.no_grad():
        p = torch.sigmoid(m(_prep_xray(path)))[0].tolist()
    return {XRAY_CLASSES[0]: p[0], XRAY_CLASSES[1]: p[1]}


def _predict_echo(path):
    import torch

    m = _load("echo")
    with torch.no_grad():
        p = torch.softmax(m(_prep_echo(path)).float(), 1)[0].tolist()
    return dict(zip(ECHO_CLASSES, p))


def _predict_mri(files):
    import torch

    m = _load("mri")
    x, ed_name, es_name = _prep_mri(files)
    with torch.no_grad():
        p = torch.softmax(m(x).float(), 1)[0].tolist()
    return dict(zip(MRI_CLASSES, p)), ed_name, es_name


def _predict_fusion(xray_path, echo_path, mri_files):
    import torch

    bundles, fm = _load("fusion")
    dev = _device()
    embeds = [None, None, None]
    present = []
    if xray_path:
        embeds[0] = bundles["xray"].encoder(_prep_xray(xray_path).to(dev))
        present.append("xray")
    if echo_path:
        embeds[1] = bundles["echo"].encoder(_prep_echo(echo_path).to(dev))
        present.append("echo")
    if mri_files:
        x, _, _ = _prep_mri(mri_files)
        embeds[2] = bundles["mri"].encoder(x.to(dev))
        present.append("mri")
    mask = fm.mask_for(present, 1, dev)
    with torch.no_grad():
        out, _ = fm(embeds, mask)
    xr = torch.sigmoid(out["xray"])[0].tolist()
    ec = torch.softmax(out["echo"].float(), 1)[0].tolist()
    mr = torch.softmax(out["mri"].float(), 1)[0].tolist()
    return present, {XRAY_CLASSES[0]: xr[0], XRAY_CLASSES[1]: xr[1]}, \
        dict(zip(ECHO_CLASSES, ec)), dict(zip(MRI_CLASSES, mr))


# --------------------------------------------------------------------------- #
# Tab handlers  (never raise — always return a friendly message)
# --------------------------------------------------------------------------- #
def _err(msg: str) -> str:
    return f"### ⚠️ {msg}"


def run_xray(image_path):
    if not image_path:
        return {}, _err("Upload a chest X-ray image (.png / .jpg) first.")
    try:
        p = _predict_xray(image_path)
    except Exception as e:
        traceback.print_exc()
        return {}, _err(f"Could not read that as a chest X-ray image. ({type(e).__name__}: {e})")
    a, b = p["Cardiomegaly"], p["Effusion"]

    def line(name, v):
        side = "above" if v >= 0.5 else "below"
        return f"**{name}: {v * 100:.1f}%** — {side} the 0.50 decision threshold."

    txt = (
        f"{line('Cardiomegaly', a)}  \n{line('Effusion', b)}  \n\n"
        "These are **independent** probabilities (multi-label), not a distribution.  \n\n"
        "_Model:_ DenseNet121 (ImageNet-pretrained), trained on NIH ChestX-ray14 "
        "(109,312 images, patient-level split). **Test mean AUROC 0.878** "
        "(Cardiomegaly 0.897, Effusion 0.859)."
    )
    return p, txt


def run_echo(video_path):
    if not video_path:
        return {}, _err("Upload an echocardiogram video (.avi) first.")
    try:
        p = _predict_echo(video_path)
    except Exception as e:
        traceback.print_exc()
        return {}, _err(f"Could not read that as an echo video. ({type(e).__name__}: {e})")
    top = max(p, key=p.get)
    txt = (
        f"**Predicted EF category: {top}**  ({p[top] * 100:.0f}% confidence).  \n\n"
        "Reduced = EF < 40 · Mildly Reduced = 40–54 · Normal = ≥ 55.  \n\n"
        "_Model:_ ResNet18 + bidirectional LSTM over 16 sampled frames, trained on "
        "EchoNet-Dynamic (official split). **Test macro one-vs-rest AUROC 0.802** "
        "(Reduced 0.906, Mildly Reduced 0.678, Normal 0.822). The *Mildly Reduced* "
        "band is intrinsically hard — a 15-point EF window with ~±5% measurement noise."
    )
    return p, txt


def run_mri(files):
    if not files:
        return {}, _err("Upload this patient's ED and ES frame volumes (two .nii.gz files).")
    try:
        p, ed_name, es_name = _predict_mri(files)
    except ValueError as e:
        return {}, _err(str(e))
    except Exception as e:
        traceback.print_exc()
        return {}, _err(f"Could not process those MRI files. ({type(e).__name__}: {e})")
    top = max(p, key=p.get)
    txt = (
        f"**Predicted diagnosis: {top} — {MRI_LONG[top]}**  ({p[top] * 100:.0f}%).  \n"
        f"_Using_ `{ed_name}` (ED) and `{es_name}` (ES).  \n\n"
        "⚠️ **Read with caution — documented limitation.** This model was trained on only "
        "**70 ACDC patients**; the test set is **15 patients (3 per class)** and **test "
        "macro AUROC is 0.706**, with HCM and MINF near chance. A single prediction here "
        "illustrates the pipeline — it is **not** a diagnosis. The MRI encoder's real role "
        "is as a component of the fusion model."
    )
    return p, txt


def run_fusion(xray_path, echo_path, mri_files):
    if not (xray_path or echo_path or mri_files):
        return ("", _err("Upload at least one modality (any 1, 2 or 3)."), {}, {}, {}, "")
    try:
        present, xr, ec, mr = _predict_fusion(xray_path, echo_path, mri_files)
    except ValueError as e:
        return ("", _err(str(e)), {}, {}, {}, "")
    except Exception as e:
        traceback.print_exc()
        return ("", _err(f"Fusion inference failed. ({type(e).__name__}: {e})"), {}, {}, {}, "")

    banner = ""
    if len(present) == 3:
        banner = (
            "<div class='synthetic-banner'>⚠️ SYNTHETIC EXAMPLE — NOT ONE REAL PATIENT.<br>"
            "No single patient in any available dataset has all three imaging modalities. "
            "This combines <b>three different patients'</b> real scans to demonstrate the "
            "fusion <b>architecture</b> only — it is not validated multi-modal performance.</div>"
        )
    name = {"xray": "X-ray", "echo": "ECHO", "mri": "MRI"}
    presence_md = "**This run:**  \n" + "  \n".join(
        f"- {name[k]}: " + ("✅ **real upload**" if k in present
                            else "🔀 *learned missing-modality token* (no file given)")
        for k in ["xray", "echo", "mri"]
    )
    how = (
        "**How the fusion layer combines modalities:** every present modality's *frozen* "
        "encoder produces a 1024-d embedding; every absent modality is replaced by a "
        "**learned** missing-modality token (a trained 1024-d parameter — not zeros). The "
        "three vectors are LayerNorm'd, concatenated (→ 3072), passed through a shared "
        "2-layer MLP (→ 512), then read by three task-specific heads.  \n\n"
        "_Single-modality-present validation on the real test sets:_ X-ray mean AUROC "
        "**0.865**, ECHO macro AUROC **0.806**, MRI macro AUROC **0.628** — the fusion "
        "pathway preserves each encoder's signal. No real tri-modal patient data exists, "
        "so a true multi-modal accuracy figure cannot be reported."
    )
    return banner, presence_md, xr, ec, mr, how


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_app() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}", elem_id="hdr")
        gr.Markdown(
            "Frozen per-modality encoders (chest X-ray · echo video · cardiac MRI), each a "
            "1024-d embedding, plus a late-fusion layer. Upload a file in any tab — inference "
            "runs on CPU in a few seconds."
        )
        with gr.Tabs():
            with gr.Tab("1 · Chest X-ray"):
                gr.Markdown("### Cardiomegaly & Effusion detection — NIH ChestX-ray14")
                with gr.Row():
                    with gr.Column():
                        xr_in = gr.Image(type="filepath", label="Chest X-ray (.png / .jpg)", height=340)
                        xr_btn = gr.Button("Analyze X-ray", variant="primary")
                    with gr.Column():
                        xr_out = gr.Label(label="Predicted probabilities (independent)", num_top_classes=2)
                xr_txt = gr.Markdown(elem_classes="bigtext")
                xr_btn.click(run_xray, xr_in, [xr_out, xr_txt])

            with gr.Tab("2 · Echocardiogram"):
                gr.Markdown("### Ejection-fraction category — EchoNet-Dynamic")
                with gr.Row():
                    with gr.Column():
                        ec_in = gr.Video(label="Echo video (.avi)", height=340)
                        ec_btn = gr.Button("Analyze ECHO", variant="primary")
                    with gr.Column():
                        ec_out = gr.Label(label="EF category probabilities", num_top_classes=3)
                ec_txt = gr.Markdown(elem_classes="bigtext")
                ec_btn.click(run_echo, ec_in, [ec_out, ec_txt])

            with gr.Tab("3 · Cardiac MRI"):
                gr.Markdown("### 5-class diagnosis — ACDC  ·  upload one patient's **ED + ES** frame volumes")
                with gr.Row():
                    with gr.Column():
                        mr_in = gr.File(label="ED & ES volumes (.nii.gz) — 2 files, not the _gt masks",
                                        file_count="multiple", file_types=[".gz", ".nii"])
                        mr_btn = gr.Button("Analyze MRI", variant="primary")
                    with gr.Column():
                        mr_out = gr.Label(label="Diagnosis probabilities", num_top_classes=5)
                mr_txt = gr.Markdown(elem_classes="bigtext")
                mr_btn.click(run_mri, mr_in, [mr_out, mr_txt])

            with gr.Tab("4 · Fusion  (centerpiece)"):
                gr.Markdown("### Multi-modal fusion — upload **any 1, 2, or 3** modalities")
                fu_banner = gr.HTML()
                with gr.Row():
                    fu_xray = gr.Image(type="filepath", label="X-ray (optional)", height=220)
                    fu_echo = gr.Video(label="ECHO (optional)", height=220)
                    fu_mri = gr.File(label="MRI ED+ES (optional)", file_count="multiple",
                                     file_types=[".gz", ".nii"])
                fu_btn = gr.Button("Run fusion", variant="primary", size="lg")
                fu_presence = gr.Markdown(elem_classes="bigtext")
                with gr.Row():
                    fu_xray_out = gr.Label(label="X-ray head", num_top_classes=2)
                    fu_echo_out = gr.Label(label="EF head", num_top_classes=3)
                    fu_mri_out = gr.Label(label="Diagnosis head", num_top_classes=5)
                fu_how = gr.Markdown(elem_classes="bigtext")
                fu_btn.click(run_fusion, [fu_xray, fu_echo, fu_mri],
                             [fu_banner, fu_presence, fu_xray_out, fu_echo_out, fu_mri_out, fu_how])

        gr.Markdown(
            "---\n_Capstone project. The three encoders were trained on three separate "
            "public datasets with **no shared patients**; the Fusion tab's 3-modality mode "
            "is a synthetic architecture demonstration, flagged with a red banner whenever "
            "it occurs._"
        )
    return demo


if __name__ == "__main__":
    build_app().queue().launch(
        server_name="127.0.0.1", server_port=7860, show_error=True,
        css=BIG_CSS, theme=gr.themes.Soft(),
    )
