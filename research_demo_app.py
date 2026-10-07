"""Faculty-facing, read-only Gradio presentation for the frozen X-ray study.

Launch with ``python research_demo_app.py`` from the repository root.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# The bundled image pipeline is already version-pinned in the environment; avoid
# Albumentations' optional update check so the demo remains offline-capable.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import gradio as gr
import pandas as pd

from src import config as C
from src.ui_data import LABELS, load_research_data, temperature_probabilities

TITLE = "Separating Calibration and Threshold Effects in Multi-Label Chest X-Ray Classification"
DATA: dict[str, Any] | None = None

CSS = """
body, .gradio-container { background:#f4f6f8 !important; color:#18232d !important; }
.gradio-container { max-width:1280px !important; }
h1,h2,h3 { color:#172c3d !important; }
.lead { border-left:4px solid #27707a; padding:12px 18px; background:#eaf1f2; }
.warning { border-left:4px solid #9a5b28; padding:12px 18px; background:#fbf2e8; }
"""


def _data() -> dict[str, Any]:
    global DATA
    if DATA is None:
        DATA = load_research_data()
    return DATA


def _source(path: str) -> str:
    return f"<details><summary>Source artifacts</summary><small><code>{path}</code></small></details>"


def _md_table(df: pd.DataFrame, cols: list[str], digits: int = 4) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    out = df[present].copy()
    for col in out.select_dtypes(include="number"):
        out[col] = out[col].round(digits)
    return out


def _figure(name: str):
    try:
        p = _data()["figures"][name]
        return str(p) if Path(p).is_file() else None
    except (FileNotFoundError, KeyError):
        return None


def overview(mode: str):
    try:
        d = _data()
        m = d["manifest"]
        details = "" if mode == "Presentation" else (
            f"\n\n{_source('outputs/final_research_manifest.json')}  "
            f"\nCheckpoint: `outputs/checkpoints/baseline/densenet121_best.pt` · seed {m.get('seed', 42)}"
        )
        return ("## Research question\nIn a fixed DenseNet121 classifier, how do per-label calibration and validation-frozen threshold selection separately and jointly affect calibration metrics and operating-point performance under class imbalance?\n\n"
                "**Study design** NIH ChestX-ray14 · Cardiomegaly and Effusion · DenseNet121 · 224 × 224 · patient-level split · seed 42 · thresholds fit on validation only.\n\n"
                "**Pipeline**\n\n`Chest X-ray → DenseNet121 → raw probabilities → temperature scaling → validation-frozen threshold → predicted labels → statistical evidence`\n\n"
                "**Model performance** is summarized with threshold-free AUROC. **Decision-policy performance** describes precision, recall, specificity, and F1 at a frozen operating point. These answer different questions.\n\n"
                "<div class='warning'><b>Research demonstration only.</b> This system is not intended for clinical diagnosis or treatment decisions. External validation is pending; no external metrics have been computed.</div>" + details)
    except Exception as e:
        return f"## Research artifacts unavailable\n\n{e}"


def calibration_view():
    try:
        d = _data(); cal = d["calibration"]; temps = d["temperatures"]["metadata"]["temperatures"]
        rows=[]
        for label in LABELS:
            raw=cal["raw"][label]; scaled=cal["calibrated"][label]
            rows.append({"Label":label,"T":temps[label],"Raw NLL":raw["nll"],"Scaled NLL":scaled["nll"],"Raw Brier":raw["brier"],"Scaled Brier":scaled["brier"],"Raw ECE":raw["ece"],"Scaled ECE":scaled["ece"]})
        log=d["logistic"]["val_report"]
        logdf=pd.DataFrame([{"Label":l,"NLL":log[l]["nll"],"Brier":log[l]["brier"],"ECE":log[l]["ece"]} for l in LABELS])
        return pd.DataFrame(rows).round(4), logdf.round(4), _figure("calibration"), _figure("ece"), (
          "Temperature scaling lowers NLL and Brier on validation, while reference ECE rises for both labels. Calibration quality is metric-dependent. Logistic/Platt calibration is an extension analysis, outside primary A/B/C/D. ECE sensitivity is loaded from the existing 10/15/20-bin, equal-width/equal-frequency artifact.\n\n" + _source("outputs/metrics/calibration/calibration_report_val.json; outputs/metrics/ece_sensitivity/ece_sensitivity.json"))
    except Exception as e: return pd.DataFrame(),pd.DataFrame(),None,None,f"Research artifacts unavailable: {e}"


def abcd_view():
    try:
        d=_data(); df=d["policy"]
        subset=df[(df["policy"].isin(["fixed","f1_optimal"])) & (df["variant"].isin(["raw","calibrated"]))]
        # A/B use fixed threshold; C/D use validation F1-optimal thresholds.
        mapping={("raw","fixed"):"A",("calibrated","fixed"):"B",("raw","f1_optimal"):"C",("calibrated","f1_optimal"):"D"}
        subset=subset.copy(); subset["Arm"]=[mapping[(r.variant,r.policy)] for r in subset.itertuples()]
        view=_md_table(subset,["class","Arm","threshold","test_f1","test_precision","test_recall","test_specificity"])
        view=view.rename(columns={"class":"Label","threshold":"Frozen threshold","test_f1":"Test F1","test_precision":"Test precision","test_recall":"Test recall","test_specificity":"Test specificity"}).sort_values(["Label","Arm"])
        return view, "**Interpretation:** the model and raw scores are fixed; arms vary only the calibration transform and threshold policy. Thresholds are selected on validation and then evaluated unchanged on test. Monotonic temperature scaling preserves ranking, so paired raw/calibrated operating points can share threshold-free AUROC while decision values change.\n\n"+_source("outputs/final_results/decision_policy_analysis.csv; outputs/metrics/thresholds/thresholds_val.json; outputs/metrics/thresholds/thresholds_calibrated_val.json")
    except Exception as e:return pd.DataFrame(),f"Research artifacts unavailable: {e}"


def stats_view():
    try:
        d=_data(); tests=d["paired_tests"]
        ci=d["arm_differences"]
        f=ci[(ci["split"]=="test")&(ci["arm_a"]=="A")&(ci["arm_b"]=="D")&(ci["metric"]=="f1")]
        au=d["delong"]["per_label"]
        primary=[r for r in tests["per_label_results"] if r.get("primary_hypothesis")]
        summary="; ".join(f"{r['label']}: ΔF1 {r['delta_observed']:+.4f}, permutation p={r['p_value']:.4g}, Holm p={r['holm_adjusted_p']:.4g}" for r in primary)
        return _md_table(f,["class","delta_point_estimate","ci_lower","ci_upper","n_bootstraps","method"]), pd.DataFrame([{"Label":l,"Test AUROC":au[l]["auc"],"CI low":au[l]["ci_lower"],"CI high":au[l]["ci_upper"]} for l in LABELS]).round(4), summary, _source("outputs/final_results/arm_differences.csv; outputs/metrics/patient_stats/paired_tests.json; outputs/metrics/patient_stats/delong_auroc.json")
    except Exception as e:return pd.DataFrame(),pd.DataFrame(),"",f"Research artifacts unavailable: {e}"


def stability_view():
    try:
        s=_data()["stability"]; rows=[]
        for label, variants in s["per_label"].items():
            for variant, policies in variants.items():
                p=policies.get("f1_optimal")
                if p: rows.append({"Label":label,"Variant":variant,"First fit":p["first_fit"],"Mean":p["mean"],"SD":p["sd"],"95% CI":f"[{p['ci_lower']:.4f}, {p['ci_upper']:.4f}]","Within ±5%":p["within_pm05pct_of_first_fit"],"Within ±10%":p["within_pm10pct_of_first_fit"]})
        return pd.DataFrame(rows), _figure("stability"), f"Patient-cluster bootstrap on validation: {s['n_bootstraps']:,} resamples. Vectorized versus frozen equality: {s['equivalence_check']['passed']} ({s['equivalence_check']['detail']}).\n\n{_source('outputs/metrics/threshold_stability/threshold_stability.json')}"
    except Exception as e:return pd.DataFrame(),None,f"Research artifacts unavailable: {e}"


def error_view():
    try:
        d=_data(); df=d["error"]
        return _md_table(df,["class","error_type","confidence_bin","count","percentage"]),_figure("error"),_figure("gradcam"),"Test-set error strata and the existing Grad-CAM figure are shown as exploratory analysis. Attribution is not a validated clinical localization.\n\n"+_source("outputs/final_results/error_analysis.csv; outputs/paper/figures/figure_08_error_analysis.png; outputs/paper/figures/figure_09_gradcam.png")
    except Exception as e:return pd.DataFrame(),None,None,f"Research artifacts unavailable: {e}"


def policy_view():
    try:
        d=_data(); df=d["policy"]
        return _md_table(df,["class","variant","policy","threshold","test_precision","test_recall","test_specificity","test_f1"]),_figure("policy"),_source("outputs/final_results/decision_policy_analysis.csv; outputs/paper/figures/figure_13_decision_policy.png")
    except Exception as e:return pd.DataFrame(),None,f"Research artifacts unavailable: {e}"


def prevalence_view():
    try:
        d=_data(); p=d["prevalence"]
        rows=[]
        for label, detail in p.get("per_label",{}).items():
            for cohort in detail.get("cohorts",[]): rows.append({"Label":label,**cohort})
        df=pd.DataFrame(rows)
        return df,_figure("prevalence"),str(p.get("method","Below-natural targets can be infeasible under the documented sampling design."))+"\n\n"+_source("outputs/metrics/prevalence/prevalence_shift.json; outputs/paper/figures/figure_14_prevalence_sensitivity.png")
    except Exception as e:return pd.DataFrame(),None,f"Research artifacts unavailable: {e}"


def reproducibility_view():
    try:
        d=_data(); m=d["manifest"]; ext=d["external"]
        ckpt=C.BASELINE_CHECKPOINT_BEST
        return (f"**Checkpoint:** {'available' if ckpt.is_file() else 'unavailable'} (`outputs/checkpoints/baseline/densenet121_best.pt`)  \n"
                f"**Frozen baseline:** {m.get('frozen_baseline')} · **seed:** {m.get('seed')} · **verified claims:** {m.get('claims_verified')}  \n"
                f"**External validation:** {ext.get('status')} — no external metrics computed.  \n"
                "**Reproducibility:** `python -m src.paper_build` rebuilds paper artifacts; do not run as part of demo launch. Model output and research artifact views are separated.\n\n"
                "Limitations include a single patient-level split, label prevalence effects, metric-dependent calibration behavior, and unavailable external CheXpert data. All thresholds were fitted on validation only.\n\n"+_source("outputs/final_research_manifest.json; outputs/metrics/external/status.json; docs/REPRODUCIBILITY.md"))
    except Exception as e:return f"Research artifacts unavailable: {e}"


def live_predict(image_path):
    if not image_path: return pd.DataFrame(), "Upload a chest X-ray image to begin."
    try:
        import numpy as np
        import torch
        from PIL import Image
        from src.dataset import build_transforms
        from src.inference import load_model
        d=_data(); ckpt=C.BASELINE_CHECKPOINT_BEST
        if not ckpt.is_file(): return pd.DataFrame(),"Live model checkpoint unavailable; verified research artifacts remain available."
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, labels=load_model(ckpt,device)
        if tuple(labels)!=LABELS: raise ValueError(f"Checkpoint labels {labels} do not match frozen order {LABELS}")
        image=np.asarray(Image.open(image_path).convert("RGB"))
        tensor=build_transforms(train=False,image_size=224)(image=image)["image"].unsqueeze(0).to(device)
        model.eval()
        with torch.inference_mode(): logits=model(tensor).float().cpu().flatten().tolist()
        raw=[1/(1+np.exp(-z)) for z in logits]
        t=d["temperatures"]["temperatures"]
        calibrated=temperature_probabilities(logits,t)
        thresholds=d["thresholds_cal"]["policies"]["f1_optimal"]
        rows=[]
        for i,label in enumerate(LABELS):
            threshold=float(thresholds[label]["threshold"])
            rows.append({"Label":label,"Raw probability":float(raw[i]),"Calibrated score":calibrated[i],"Frozen threshold":threshold,"Predicted label":"positive" if calibrated[i]>=threshold else "negative"})
        return pd.DataFrame(rows).round(4),"Model output demonstration only; this is not a diagnosis. Thresholds are the validation-frozen calibrated F1-optimal policy."
    except Exception as e:
        return pd.DataFrame(),f"The image could not be processed: {type(e).__name__}: {e}"


def build_app():
    with gr.Blocks(title=TITLE) as app:
        gr.Markdown(f"# {TITLE}\n### Controlled DenseNet121 Study on Cardiomegaly and Effusion")
        mode=gr.Radio(["Presentation","Technical"],value="Presentation",label="Display mode")
        with gr.Tabs():
            with gr.Tab("Research Overview"):
                ov=gr.Markdown(overview("Presentation"),elem_classes="lead")
                mode.change(overview,mode,ov)
            with gr.Tab("Live X-ray Demonstration"):
                gr.Markdown("Upload an image for fixed-checkpoint inference. Outputs are model scores and policy labels, never diagnoses.")
                with gr.Row():
                    image=gr.Image(type="filepath",label="Chest X-ray",height=330)
                    result=gr.Dataframe(label="Per-label pipeline output",interactive=False)
                sample_dir=C.PROJECT_ROOT / "demo_samples" / "xray"
                samples=sorted(sample_dir.glob("*.png")) if sample_dir.is_dir() else []
                if samples:
                    gr.Examples(examples=[[str(p)] for p in samples],inputs=image,label="Curated test-set examples")
                msg=gr.Markdown(); gr.Button("Run frozen pipeline",variant="primary").click(live_predict,image,[result,msg])
            with gr.Tab("Calibration"):
                cal=gr.Dataframe(label="Validation temperature calibration"); log=gr.Dataframe(label="Extension: logistic / Platt calibration")
                cf=gr.Image(label="Calibration comparison"); ef=gr.Image(label="ECE sensitivity (existing figure)"); cm=gr.Markdown()
                app.load(calibration_view,None,[cal,log,cf,ef,cm])
            with gr.Tab("A/B/C/D Decision Policy"):
                gr.Markdown("| Arm | Calibration | Threshold |\n|---|---|---|\n| A | None | Fixed 0.50 |\n| B | Temperature | Fixed 0.50 |\n| C | None | Validation F1-optimal |\n| D | Temperature | Validation F1-optimal |")
                ab=gr.Dataframe(label="Test operating point results"); abm=gr.Markdown(); app.load(abcd_view,None,[ab,abm])
            with gr.Tab("Statistical Evidence"):
                st=gr.Dataframe(label="Patient-level bootstrap D−A F1 differences"); au=gr.Dataframe(label="Test AUROC with DeLong CI"); sm=gr.Markdown()
                app.load(stats_view,None,[st,au,gr.Textbox(visible=False),sm])
            with gr.Tab("Threshold Stability"):
                ts=gr.Dataframe(label="Validation patient-bootstrap F1-optimal thresholds"); tf=gr.Image(); tm=gr.Markdown(); app.load(stability_view,None,[ts,tf,tm])
            with gr.Tab("Error Analysis"):
                er=gr.Dataframe(label="Test error strata"); erf=gr.Image(label="Error analysis figure"); gc=gr.Image(label="Exploratory Grad-CAM artifact"); erm=gr.Markdown(); app.load(error_view,None,[er,erf,gc,erm])
            with gr.Tab("Decision Policy Comparison"):
                pol=gr.Dataframe(label="Test decision-policy metrics"); pf=gr.Image(); pm=gr.Markdown(); app.load(policy_view,None,[pol,pf,pm])
            with gr.Tab("Prevalence Sensitivity"):
                pv=gr.Dataframe(label="Existing prevalence-shift scenarios"); pvf=gr.Image(); pvm=gr.Markdown(); app.load(prevalence_view,None,[pv,pvf,pvm])
            with gr.Tab("Reproducibility & Limitations"):
                gr.Markdown(reproducibility_view())
        gr.Markdown("Research demonstration only · not for clinical use · External validation pending")
    return app


if __name__ == "__main__":
    build_app().queue().launch(server_name="127.0.0.1",server_port=int(os.getenv("GRADIO_SERVER_PORT","7860")),show_error=False,theme=gr.themes.Base(),css=CSS)
