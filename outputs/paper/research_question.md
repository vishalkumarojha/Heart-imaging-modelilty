# Research Question

**Question**

> Can calibration-aware, class-specific decision policies improve the operating
> characteristics of a fixed DenseNet121 multi-label chest X-ray classifier
> under class imbalance?

This phase of the study does **not** claim that the question above has been
answered for clinical deployment. The question is answered only at the level of
controlled, retrospective offline experimentation on a single public dataset
(NIH ChestX-ray14), with external validation explicitly pending.

**Operationalised answer (see `draft_results.md`, `draft_discussion.md`)**

Yes, under the conditions tested:

- A validation-frozen, class-specific **threshold policy** (F1-optimal) raised
  test F1 for Cardiomegaly from 0.2836 to 0.3536 and for Effusion from 0.4676
  to 0.4866, with corresponding precision gains (0.1787 to 0.3377 and 0.3320 to
  0.4440).
- Per-label **temperature scaling** improved NLL and Brier but increased ECE in
  this implementation (see Section C, "Calibration interpretation"), and did
  not independently explain the F1 improvement.
- The observed gains are therefore attributed primarily to threshold
  selection rather than to calibration.

**Evidence**

- `outputs/paper/paper_evidence_map.json` — every claim mapped to a source.
- `outputs/paper/claim_audit.csv` — every claim verified against its artifact.
- `outputs/final_results/master_results.csv` — the canonical numbers.
- `outputs/final_results/verification_report.json` — 25/25 doc-vs-artifact PASS.

**Boundary**

Clinical deployment, diagnostic accuracy in practice, cross-dataset
generalisation, and prospective evaluation are NOT claimed.