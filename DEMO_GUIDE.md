# Research Demonstration Dashboard

This is a faculty-facing presentation layer for the frozen NIH ChestX-ray14
calibration and decision-policy study. It reads verified artifacts and does not
train, refit, or alter research results.

## Install and launch

From the repository root, using the project environment:

```bash
.venv/bin/python research_demo_app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). To use another local port,
set `GRADIO_SERVER_PORT`, for example `GRADIO_SERVER_PORT=7861
.venv/bin/python research_demo_app.py`. For a fresh environment, create a Python
virtual environment, install the project requirements and `requirements_demo.txt`,
then launch as above. Gradio is served locally; the dashboard does not require
GitHub or an external service.

## Checkpoint and sample image

Live inference uses `outputs/checkpoints/baseline/densenet121_best.pt` and the
exact evaluation transform from `src.dataset.build_transforms`: RGB conversion,
224 × 224 resize, ImageNet normalization, tensor conversion. The checkpoint labels
must be ordered Cardiomegaly, Effusion. The output path is raw logits → sigmoid
probabilities → frozen per-label temperature scaling `sigmoid(z/T)` → calibrated
validation-frozen F1-optimal thresholds. The research code and threshold fitting
are not invoked during inference. If the checkpoint is missing, live inference
reports that it is unavailable; artifact tabs continue to work.

Use a PNG or JPEG chest X-ray. Curated examples, if present, are described in
[`demo_samples/README.md`](demo_samples/README.md). An uploaded image receives
model scores and predicted labels, not a diagnosis. No live Grad-CAM is created;
the dashboard may show the saved, exploratory research Grad-CAM figure with its
nonclinical attribution caveat.

## Dashboard tabs

1. **Research Overview** — question, fixed model and data design, pipeline, and
   research-only notice.
2. **Live X-ray Demonstration** — raw probability, calibrated score, frozen
   threshold, and per-label model output for an uploaded image.
3. **Calibration** — validation NLL, Brier score, ECE, temperature values, the
   separately marked logistic extension, and the existing ECE-sensitivity plot.
4. **A/B/C/D Decision Policy** — A: raw + 0.50; B: temperature + 0.50; C: raw +
   validation F1-optimal threshold; D: temperature + validation F1-optimal
   threshold. Test operating-point results use those frozen thresholds.
5. **Statistical Evidence** — patient-level bootstrap D−A F1 differences and
   intervals, paired permutation/Holm results, and test AUROC with DeLong CIs.
6. **Threshold Stability** — validation patient-cluster bootstrap summaries and
   the recorded vectorized/frozen equality check.
7. **Error Analysis** — saved test error strata, error figure, and exploratory
   Grad-CAM figure. Attribution is not validated clinical localization.
8. **Decision Policy Comparison** — test precision, recall, specificity, and F1
   across frozen operating policies, distinct from threshold-free model AUROC.
9. **Prevalence Sensitivity** — existing negative-subsampling simulation and
   feasibility notes; it is not external validation.
10. **Reproducibility & Limitations** — manifest, checkpoint availability,
    validation-only fitting, study limitations, and external-data status.

## Presentation flow

Select **Presentation** mode, then move through Overview → Live X-ray → A/B/C/D
→ Statistical Evidence → Threshold Stability → Error Analysis → Limitations.
Select **Technical** mode to expose source-path and audit details. Artifact tables
and figures point to repository-generated results rather than copied metric values.

## Limitations and external validation

The dataset split and DenseNet121 baseline are fixed. Thresholds and temperature
parameters were fit on validation data only; the held-out test set is used only
for evaluation. Temperature scaling improves NLL and Brier in the reported
validation artifacts while reference ECE rises for both labels, illustrating
metric-dependent calibration behavior. Results come from a single patient-level
split. External CheXpert validation is **PENDING / BLOCKED** because the dataset
is unavailable; no external metrics were computed. This dashboard is a research
demonstration, not clinical software.

## Existing multimodal demo

The earlier X-ray / echocardiogram / MRI / fusion demo remains available with
`python demo_app.py`. It is separate from this calibration-focused research
dashboard and retains its synthetic multi-modality caveats.
