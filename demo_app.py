"""Multi-Modal Cardiac Condition Detection System — Capstone Demo (Gradio app).

    python demo_app.py            # -> http://127.0.0.1:7860

4 tabs: X-ray, Echocardiogram, MRI, Fusion. Upload a file, get a live prediction
from the frozen Phase 1-3 encoders + their trained heads, and (tab 4) the Phase 4
fusion pathway. All inference runs on CPU (~1-3 s per prediction).

Real / synthetic transparency: the Fusion tab shows a prominent banner whenever
all three modalities are supplied together, because no real patient in any of the
three source datasets has more than one modality.
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

# --------------------------------------------------------------------------- #
# Look & feel — a deliberate clinical palette, two typefaces, flat surfaces.
#   ground  #f5f6f8 cool paper (not cream)   ink   #1b1f24
#   muted   #59616c                          line  #dde1e6
#   accent  #9c2f38 (ECG red — primary action + title rule only)
#   Display: Newsreader (roman).  Body: Public Sans.  Mono: IBM Plex Mono.
# --------------------------------------------------------------------------- #
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Public Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');

:root, .gradio-container {
  --ground:#f5f6f8; --panel:#ffffff; --ink:#1b1f24; --muted:#59616c;
  --line:#dde1e6; --accent:#9c2f38;
  --body-background-fill:var(--ground);
  --background-fill-primary:var(--panel);
  --background-fill-secondary:var(--ground);
  --block-background-fill:var(--panel);
  --block-border-color:var(--line);
  --block-border-width:1px;
  --block-shadow:none;
  --block-radius:10px;
  --border-color-primary:var(--line);
  --body-text-color:var(--ink);
  --body-text-color-subdued:var(--muted);
  --button-primary-background-fill:var(--accent);
  --button-primary-background-fill-hover:#872a31;
  --button-primary-text-color:#ffffff;
  --button-primary-border-color:var(--accent);
  --radius-lg:10px; --radius-md:8px; --radius-sm:6px;
}
.gradio-container {max-width:1080px !important; margin:0 auto; background:var(--ground);}
.gradio-container, .gradio-container .prose {color:var(--ink); font-size:16px; line-height:1.6;}

/* headings: Newsreader roman, real size steps, no tracking */
.gradio-container h1,.gradio-container h2,.gradio-container h3,.gradio-container h4 {
  font-family:'Newsreader',Georgia,'Times New Roman',serif;
  font-weight:500; letter-spacing:0; color:var(--ink); line-height:1.2;
}
.gradio-container h3 {font-size:1.32rem; margin:.2rem 0 .9rem;}
.gradio-container .prose p {max-width:74ch; color:var(--ink);}
.gradio-container .prose em {color:var(--muted); font-style:normal;}
.gradio-container .prose code {font-size:.92em; background:#eef0f3; padding:.08em .35em; border-radius:4px;}

/* app header */
.app-head {padding:26px 4px 10px;}
.app-title {font-family:'Newsreader',Georgia,serif; font-weight:500; font-size:2.15rem;
  line-height:1.12; margin:0; color:var(--ink);}
.app-title .rule {display:block; width:52px; height:3px; background:var(--accent); margin:14px 0 0;}
.app-sub {margin:12px 0 0; color:var(--muted); font-size:1rem; max-width:70ch;}

/* flat panels — one hairline edge, no shadow, modest radius */
.gradio-container .block, .gradio-container .form {box-shadow:none !important; border-radius:10px;}
.gradio-container .tab-nav button {font-size:.98rem; letter-spacing:0;}
.gradio-container .tab-nav button.selected {color:var(--accent); border-bottom-color:var(--accent);}

/* result readout */
.readout .prose p {font-size:1.02rem;}
.marks {font-variant-numeric:tabular-nums;}
.marks .on {color:var(--accent); font-weight:600;}
.marks .off {color:var(--muted);}

/* synthetic-example banner: a solid block with a heavy left rule, not an alert toast */
.synthetic-banner {
  background:#fbeced; border:1px solid #e6b9bd; border-left:6px solid var(--accent);
  color:#4d1a1f; padding:14px 16px; border-radius:8px; margin:4px 0 2px;
  font-size:1rem; line-height:1.5;
}
.synthetic-banner b {font-weight:700;}

.app-foot {margin-top:22px; padding-top:14px; border-top:1px solid var(--line);
  color:var(--muted); font-size:.92rem; max-width:78ch;}
"""

# --------------------------------------------------------------------------- #
# Lazy model cache
# --------------------------------------------------------------------------- #
_CACHE: dict = {}


def _device():
    import torch

    return torch.device("cpu")


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
# Preprocessing (reuses the exact Phase 1-3 pipelines)
# --------------------------------------------------------------------------- #
def _prep_xray(path):
    import numpy as np
    from PIL import Image

    from src.dataset import build_transforms

    img = np.array(Image.open(path).convert("RGB"))
    return build_transforms(train=False, image_size=224)(image=img)["image"].unsqueeze(0)


def _prep_echo(path):
    from src.echo_dataset import apply_clip, build_transforms, load_norm_stats, read_clip

    frames = read_clip(str(path), 16)
    mean, std = load_norm_stats()
    return apply_clip(build_transforms(False, 112, mean, std), frames).unsqueeze(0)


def _resolve_ed_es(files):
    paths = [str(getattr(f, "name", f)) for f in files]
    vols = [p for p in paths if p.endswith((".nii.gz", ".nii")) and "_gt" not in os.path.basename(p)]
    if len(vols) < 2:
        raise ValueError(
            f"Need the two frame volumes (.nii.gz) for one patient. Got {len(vols)} usable "
            f"of {len(paths)} uploaded. Do not include the *_gt.nii.gz masks."
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
# Predict
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
    return (present, {XRAY_CLASSES[0]: xr[0], XRAY_CLASSES[1]: xr[1]},
            dict(zip(ECHO_CLASSES, ec)), dict(zip(MRI_CLASSES, mr)))


# --------------------------------------------------------------------------- #
# Tab handlers (never raise)
# --------------------------------------------------------------------------- #
def _err(msg: str) -> str:
    return f"### {msg}"


def run_xray(image_path):
    if not image_path:
        return {}, _err("Upload a chest X-ray image (PNG or JPEG) to begin.")
    try:
        p = _predict_xray(image_path)
    except Exception as e:
        traceback.print_exc()
        return {}, _err(f"That file could not be read as a chest X-ray. ({type(e).__name__})")
    a, b = p["Cardiomegaly"], p["Effusion"]

    def line(name, v):
        return f"**{name}: {v * 100:.1f}%**, {'above' if v >= 0.5 else 'below'} the 0.50 decision threshold."

    txt = (
        f"{line('Cardiomegaly', a)}  \n{line('Effusion', b)}  \n\n"
        "The two probabilities are scored independently (multi-label); they are not "
        "one distribution and need not sum to 100%.  \n\n"
        "_Model: DenseNet121, ImageNet-pretrained, trained on NIH ChestX-ray14 "
        "(109,312 images, patient-level split). Test mean AUROC 0.878; per label, "
        "Cardiomegaly 0.897 and Effusion 0.859. AUROC is the reported metric because "
        "Cardiomegaly appears in about 2.5% of images._"
    )
    return p, txt


def run_echo(video_path):
    if not video_path:
        return {}, _err("Upload an echocardiogram video (AVI) to begin.")
    try:
        p = _predict_echo(video_path)
    except Exception as e:
        traceback.print_exc()
        return {}, _err(f"That file could not be read as an echo video. ({type(e).__name__})")
    top = max(p, key=p.get)
    txt = (
        f"**Predicted ejection-fraction category: {top}**, at {p[top] * 100:.0f}% confidence.  \n\n"
        "Category boundaries: Reduced below 40, Mildly Reduced 40 to 54, Normal 55 and above.  \n\n"
        "_Model: ResNet18 applied per frame with a bidirectional LSTM over 16 sampled "
        "frames, trained on EchoNet-Dynamic (official split). Test macro one-vs-rest "
        "AUROC 0.802; by class, Reduced 0.906, Normal 0.822, Mildly Reduced 0.678. "
        "The middle band is hard by construction: a 15-point window, and echo-derived "
        "EF itself carries roughly ±5% measurement noise._"
    )
    return p, txt


def run_mri(files):
    if not files:
        return {}, _err("Upload one patient's ED and ES frame volumes (two .nii.gz files).")
    try:
        p, ed_name, es_name = _predict_mri(files)
    except ValueError as e:
        return {}, _err(str(e))
    except Exception as e:
        traceback.print_exc()
        return {}, _err(f"Those MRI files could not be processed. ({type(e).__name__})")
    top = max(p, key=p.get)
    txt = (
        f"**Predicted diagnosis: {top}, {MRI_LONG[top]}**, at {p[top] * 100:.0f}% confidence.  \n"
        f"_Read from_ `{ed_name}` _(ED) and_ `{es_name}` _(ES)._  \n\n"
        "**Read this prediction with caution.** It is a documented limitation. The model "
        "was trained on 70 ACDC patients; the test set holds 15 patients, three per class; "
        "test macro AUROC is 0.706, and HCM and MINF sit close to chance. A single "
        "prediction here shows that the pipeline runs, not that a diagnosis is reliable. "
        "The MRI encoder's real role is as one input to the fusion model.  \n\n"
        "_Classes: DCM dilated cardiomyopathy, HCM hypertrophic cardiomyopathy, "
        "MINF prior myocardial infarction, NOR normal, RV abnormal right ventricle._"
    )
    return p, txt


def run_fusion(xray_path, echo_path, mri_files):
    if not (xray_path or echo_path or mri_files):
        return "", _err("Upload at least one modality. Any one, two, or all three works."), {}, {}, {}, ""
    try:
        present, xr, ec, mr = _predict_fusion(xray_path, echo_path, mri_files)
    except ValueError as e:
        return "", _err(str(e)), {}, {}, {}, ""
    except Exception as e:
        traceback.print_exc()
        return "", _err(f"Fusion inference failed. ({type(e).__name__})"), {}, {}, {}, ""

    banner = ""
    if len(present) == 3:
        banner = (
            "<div class='synthetic-banner'><b>Synthetic example. Not one real patient.</b><br>"
            "No single patient in any of these datasets has all three imaging modalities. "
            "This run combines scans from three different people to exercise the fusion "
            "architecture. It is not a validated multi-modal result.</div>"
        )
    label = {"xray": "X-ray", "echo": "Echocardiogram", "mri": "MRI"}
    presence_md = "**This run**  \n<span class='marks'>" + "  \n".join(
        (f"<span class='on'>&#9679; {label[k]}</span> — real upload" if k in present
         else f"<span class='off'>&#9675; {label[k]}</span> — substituted with the learned missing-modality token")
        for k in ["xray", "echo", "mri"]
    ) + "</span>"
    how = (
        "**How the fusion layer combines modalities.** Each present modality's frozen "
        "encoder produces a 1024-dimensional embedding. Each absent modality is replaced "
        "by a learned missing-modality token, a trained 1024-dimensional parameter rather "
        "than a zero vector. The three vectors are LayerNorm'd, concatenated to 3072, "
        "passed through a shared two-layer MLP to a 512-dimensional representation, and "
        "read by three task heads.  \n\n"
        "_Single-modality-present validation, on the real test sets: X-ray mean AUROC "
        "0.865, Echocardiogram macro AUROC 0.806, MRI macro AUROC 0.628. The fusion "
        "pathway preserves each encoder's signal. No real tri-modal patient data exists, "
        "so a true multi-modal accuracy figure is not reported._"
    )
    return banner, presence_md, xr, ec, mr, how


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
HEADER = (
    "<div class='app-head'>"
    "<h1 class='app-title'>Multi-Modal Cardiac Condition Detection"
    "<span class='rule'></span></h1>"
    "<p class='app-sub'>Three imaging modalities, three frozen encoders, one late-fusion "
    "model. A capstone demonstration: upload a file in any tab for a live prediction.</p>"
    "</div>"
)
FOOTER = (
    "<div class='app-foot'>The three encoders were trained on three separate public "
    "datasets with no shared patients. The Fusion tab's three-modality mode is an "
    "architecture demonstration on unrelated scans, marked as synthetic whenever it "
    "occurs. Inference runs on CPU.</div>"
)


def build_app() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.HTML(HEADER)

        with gr.Tabs():
            with gr.Tab("X-ray"):
                gr.Markdown("### Cardiomegaly and effusion — NIH ChestX-ray14")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        xr_in = gr.Image(type="filepath", label="Chest X-ray (PNG / JPEG)", height=360)
                        xr_btn = gr.Button("Analyze X-ray", variant="primary")
                    with gr.Column(scale=6):
                        xr_out = gr.Label(label="Probability (each scored independently)", num_top_classes=2)
                        xr_txt = gr.Markdown(elem_classes="readout")
                xr_btn.click(run_xray, xr_in, [xr_out, xr_txt])

            with gr.Tab("Echocardiogram"):
                gr.Markdown("### Ejection-fraction category — EchoNet-Dynamic")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        ec_in = gr.Video(label="Echo video (AVI)", height=360)
                        ec_btn = gr.Button("Analyze echo", variant="primary")
                    with gr.Column(scale=6):
                        ec_out = gr.Label(label="Category probability", num_top_classes=3)
                        ec_txt = gr.Markdown(elem_classes="readout")
                ec_btn.click(run_echo, ec_in, [ec_out, ec_txt])

            with gr.Tab("MRI"):
                gr.Markdown("### Five-class diagnosis — ACDC")
                gr.Markdown(
                    "Upload one patient's **ED and ES frame volumes** together "
                    "(`patientNNN_frameXX.nii.gz`). Not the `_4d` or `_gt` files.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        mr_in = gr.File(label="ED + ES volumes (.nii.gz)", file_count="multiple",
                                        file_types=[".gz", ".nii"])
                        mr_btn = gr.Button("Analyze MRI", variant="primary")
                    with gr.Column(scale=6):
                        mr_out = gr.Label(label="Diagnosis probability", num_top_classes=5)
                        mr_txt = gr.Markdown(elem_classes="readout")
                mr_btn.click(run_mri, mr_in, [mr_out, mr_txt])

            with gr.Tab("Fusion"):
                gr.Markdown("### Multi-modal fusion")
                gr.Markdown(
                    "Provide any one, two, or all three modalities. Anything you leave "
                    "empty is filled with the learned missing-modality token.")
                fu_banner = gr.HTML()
                with gr.Row(equal_height=False):
                    fu_xray = gr.Image(type="filepath", label="X-ray (optional)", height=210)
                    fu_echo = gr.Video(label="Echocardiogram (optional)", height=210)
                    fu_mri = gr.File(label="MRI ED + ES (optional)", file_count="multiple",
                                     file_types=[".gz", ".nii"])
                fu_btn = gr.Button("Run fusion", variant="primary", size="lg")
                fu_presence = gr.Markdown(elem_classes="readout")
                with gr.Row(equal_height=True):
                    fu_xray_out = gr.Label(label="X-ray head", num_top_classes=2)
                    fu_echo_out = gr.Label(label="Ejection-fraction head", num_top_classes=3)
                    fu_mri_out = gr.Label(label="Diagnosis head", num_top_classes=5)
                fu_how = gr.Markdown(elem_classes="readout")
                fu_btn.click(run_fusion, [fu_xray, fu_echo, fu_mri],
                             [fu_banner, fu_presence, fu_xray_out, fu_echo_out, fu_mri_out, fu_how])

        gr.HTML(FOOTER)
    return demo


if __name__ == "__main__":
    build_app().queue().launch(
        server_name="127.0.0.1", server_port=7860, show_error=True, theme=THEME, css=CSS,
    )
