"""Tests for the paper-ready evidence pack under outputs/final_results/.

These verify that the derived evidence pack (master table, confidence
intervals, tables, figures, error-analysis CSV, region analysis, Grad-CAM
export, research summary, research snapshot) exists and is internally
consistent — every point estimate for a threshold metric must equal the
master table value exactly, because both derive from the same frozen data.

Run:  python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C


def load_json(path) -> dict:
    return json.loads(Path(path).read_text())


class TestMasterResults(unittest.TestCase):
    def test_table_exists_and_is_complete(self):
        df = pd.read_csv(C.MASTER_RESULTS_CSV)
        self.assertEqual(len(df), 24)
        self.assertTrue(df["experiment_id"].isin(["A", "B", "C", "D"]).all())
        for arm, n_pol in (("A", 1), ("B", 1), ("C", 5), ("D", 5)):
            pols = df[df.experiment_id == arm]["threshold_policy"].nunique()
            self.assertEqual(pols, n_pol, f"arm {arm} policy count")

    def test_no_missing_primary_metrics(self):
        df = pd.read_csv(C.MASTER_RESULTS_CSV)
        for col in ("auroc", "auprc", "f1", "ece", "brier_score",
                    "n_samples", "n_positive", "n_negative"):
            self.assertFalse(df[col].isna().any(), f"NA leaked into {col}")
        self.assertEqual(int(df["n_samples"].iloc[0]), 15884)

    def test_auroc_matches_exp1_raw(self):
        df = pd.read_csv(C.MASTER_RESULTS_CSV)
        e1 = load_json(C.EXPERIMENT_METRICS_DIR / "exp1_calibration.json")
        for _, r in df[df.experiment_id == "A"].iterrows():
            art = e1["splits"]["test"][r["class"]]["auroc_raw"]
            self.assertAlmostEqual(r["auroc"], art, places=4)


class TestConfidenceIntervals(unittest.TestCase):
    def test_table_exists_and_covers_all_metrics(self):
        df = pd.read_csv(C.CONFIDENCE_INTERVALS_CSV)
        self.assertEqual(len(df), 112)
        for metric in ("auroc", "auprc", "accuracy", "precision", "recall",
                       "specificity", "f1", "ece", "brier_score"):
            self.assertIn(metric, set(df.metric), metric)

    def test_point_estimate_bracketed_by_ci(self):
        df = pd.read_csv(C.CONFIDENCE_INTERVALS_CSV)
        tol = 1e-4
        inside = ((df.point_estimate >= df.ci_lower - tol)
                  & (df.point_estimate <= df.ci_upper + tol))
        self.assertTrue(inside.all(), df[~inside])

    def test_threshold_metric_points_equal_master_table(self):
        ci = pd.read_csv(C.CONFIDENCE_INTERVALS_CSV)
        master = pd.read_csv(C.MASTER_RESULTS_CSV)
        variant = {"A": "raw", "B": "calibrated", "C": "raw", "D": "calibrated"}
        for _, r in master[master.experiment_id.isin(["C", "D"])].iterrows():
            row = ci[(ci["class"] == r["class"])
                     & (ci.calibration == variant[r.experiment_id])
                     & (ci.threshold_policy == r["threshold_policy"])]
            self.assertEqual(len(row), 5, msg=r.to_dict())
            for metric in ("f1", "accuracy", "precision", "recall", "specificity"):
                pe = row[row.metric == metric]["point_estimate"].iloc[0]
                self.assertAlmostEqual(pe, float(r[metric]), places=4,
                                       msg=f"{r.experiment_id} {r['class']} {metric}")


class TestTables(unittest.TestCase):
    def test_all_eight_tables_exist(self):
        for i in range(1, 9):
            p = C.FINAL_TABLES_DIR / f"table_{i}_"
            matches = list(Path(C.FINAL_TABLES_DIR).glob(f"table_{i}_*.csv"))
            self.assertEqual(len(matches), 1, f"missing/unexpected table {i}")
            df = pd.read_csv(matches[0])
            self.assertGreater(len(df), 0)

    def test_table1_matches_split_index(self):
        t = pd.read_csv(C.FINAL_TABLES_DIR / "table_1_dataset.csv")
        si = pd.read_csv(C.DATA_PROCESSED_DIR / "split_index.csv")
        total = t[t.split == "total"].iloc[0]
        self.assertEqual(int(total.images), len(si))
        self.assertEqual(int(total.patients), si["Patient ID"].nunique())
        self.assertEqual(int(total.cardiomegaly_positive), int(si.Cardiomegaly.sum()))
        self.assertEqual(int(total.effusion_positive), int(si.Effusion.sum()))

    def test_table8_is_honest_absence(self):
        t = pd.read_csv(C.FINAL_TABLES_DIR / "table_8_external_validation.csv")
        self.assertEqual(int(t["available"].iloc[0]), 0)
        self.assertTrue(int(t["no_metrics_were_computed"].iloc[0]) == 1)
        self.assertIn("BLOCKED", str(t["status"].iloc[0]))


class TestFigures(unittest.TestCase):
    def test_all_eight_figures_exist_and_nonempty(self):
        for i in range(1, 9):
            p = C.FINAL_FIGURES_DIR / f"fig_{i}"
            hits = list(Path(C.FINAL_FIGURES_DIR).glob(f"fig_{i}_*.png"))
            self.assertEqual(len(hits), 1, f"fig {i}")
            self.assertGreater(hits[0].stat().st_size, 10_000, f"fig {i} suspiciously small")


class TestErrorAnalysisCsv(unittest.TestCase):
    def test_schema_and_counts(self):
        df = pd.read_csv(C.ERROR_ANALYSIS_CSV)
        self.assertEqual(len(df), 28)
        for col in ("class", "error_type", "confidence_bin", "count", "percentage"):
            self.assertIn(col, df.columns)
        ea = load_json(C.ERROR_ANALYSIS_DIR / "error_analysis_test.json")
        fp_cardio = ea["strata"]["f1_optimal"]["Cardiomegaly"]["strata"]["FP"]["count"]
        self.assertEqual(int(df[(df["class"] == "Cardiomegaly")
                                & (df.error_type == "FP")
                                & (df.confidence_bin == "all")]["count"].iloc[0]),
                         fp_cardio)


class TestRegionAnalysis(unittest.TestCase):
    def test_matches_bbox_artifact(self):
        ra = load_json(C.REGION_ANALYSIS_JSON)
        bbox = load_json(C.GRADCAM_DIR / "bbox_localization.json")
        self.assertEqual(ra["n_cases"], bbox["n_with_box"])
        self.assertAlmostEqual(ra["aggregate"]["mean_concentration_ratio"],
                               bbox["mean_concentration_ratio"], places=4)
        self.assertAlmostEqual(ra["aggregate"]["pointing_game_accuracy"],
                               bbox["pointing_game_accuracy"], places=4)
        self.assertIn("NOT a diagnostic localization", ra["scope"])


class TestGradcamExport(unittest.TestCase):
    def test_overlays_and_metadata_exported(self):
        dst = C.FINAL_GRADCAM_DIR
        self.assertEqual(len(list(dst.glob("*.png"))), 16)
        cases = pd.read_csv(dst / "gradcam_cases.csv")
        self.assertEqual(len(cases), 16)
        for col in ("sample_id", "class", "ground_truth", "prediction",
                    "confidence", "error_type"):
            self.assertIn(col, cases.columns)
        for name in ("case_list.json", "gradcam_summary.json",
                     "bbox_localization.json"):
            self.assertTrue((dst / name).exists(), name)


class TestResearchSummary(unittest.TestCase):
    def test_required_sections_present(self):
        s = load_json(C.RESEARCH_SUMMARY_JSON)
        for key in ("research_question", "baseline_description", "proposed_method",
                    "main_metric_change", "calibration_change",
                    "error_analysis_finding", "external_validation_status",
                    "no_external_metrics_computed", "limitations", "provenance"):
            self.assertIn(key, s)

    def test_delta_f1_matches_master(self):
        s = load_json(C.RESEARCH_SUMMARY_JSON)
        master = pd.read_csv(C.MASTER_RESULTS_CSV)
        for lbl in C.TARGET_LABELS:
            a = master[(master.experiment_id == "A") & (master["class"] == lbl)
                       & (master.threshold_policy == "fixed")]["f1"].iloc[0]
            d = master[(master.experiment_id == "D") & (master["class"] == lbl)
                       & (master.threshold_policy == "f1_optimal")]["f1"].iloc[0]
            self.assertAlmostEqual(s["main_metric_change"]["per_label"][lbl]["delta_f1"],
                                   float(d - a), places=4)

    def test_external_honesty_in_summary(self):
        s = load_json(C.RESEARCH_SUMMARY_JSON)
        self.assertIn("BLOCKED", s["external_validation_status"])
        self.assertTrue(s["no_external_metrics_computed"])


class TestStatisticalUpgradeArtifacts(unittest.TestCase):
    def test_delong_auroc(self):
        d = load_json(C.PATIENT_STATS_DIR / "delong_auroc.json")
        c, e = d["per_label"]["Cardiomegaly"], d["per_label"]["Effusion"]
        self.assertAlmostEqual(c["auc"], 0.8972, places=4)
        self.assertAlmostEqual(c["ci_lower"], 0.8821, places=4)
        self.assertAlmostEqual(c["ci_upper"], 0.9124, places=4)
        self.assertEqual((c["n_pos"], c["n_neg"]), (415, 15469))
        self.assertAlmostEqual(e["auc"], 0.8587, places=4)
        self.assertAlmostEqual(e["ci_lower"], 0.8507, places=4)
        self.assertAlmostEqual(e["ci_upper"], 0.8667, places=4)
        self.assertEqual((e["n_pos"], e["n_neg"]), (1997, 13887))

    def test_paired_tests_primary_hypotheses(self):
        p = load_json(C.PATIENT_STATS_DIR / "paired_tests.json")
        self.assertEqual(p["n_permutations"], 5000)
        self.assertEqual(p["rng_seed"], 42)
        self.assertEqual(len(p["primary_hypotheses"]), 2)
        card = [r for r in p["per_label_results"]
                if r["label"] == "Cardiomegaly" and r.get("primary_hypothesis") is True][0]
        self.assertLess(card["p_value"], 0.001)
        self.assertLess(card["holm_adjusted_p"], 0.001)

    def test_patient_bootstrap_artifact_contracts(self):
        ci = pd.read_csv(C.CONFIDENCE_INTERVALS_PATIENT_CSV)
        self.assertEqual(len(ci), 172)
        self.assertEqual(int(ci.n_bootstraps.iloc[0]), 5000)
        self.assertEqual(float(ci.confidence_level.iloc[0]), 0.95)
        self.assertEqual(int(ci.rng_seed.iloc[0]), 42)
        self.assertIn("resampling unit = patient", ci.method.iloc[0])
        tol = 1e-4
        inside = ((ci.point_estimate >= ci.ci_lower - tol)
                  & (ci.point_estimate <= ci.ci_upper + tol))
        self.assertTrue(inside.all(), ci[~inside])
        d = ci[(ci["class"] == "Cardiomegaly") & (ci.calibration == "calibrated")
               & (ci.threshold_policy == "f1_optimal") & (ci.metric == "f1")].iloc[0]
        self.assertAlmostEqual(d.point_estimate, 0.3536, places=4)
        ciw = d.ci_upper - d.ci_lower
        self.assertAlmostEqual(ciw, 0.1361, places=4)

    def test_arm_differences_and_primary_endpoints(self):
        ad = pd.read_csv(C.ARM_DIFFERENCES_CSV)
        self.assertEqual(len(ad), 40)
        rep = load_json(C.STATISTICAL_REPORT_JSON)
        for lbl, exp_d, exp_lo, exp_hi in [
                ("Cardiomegaly", 0.0700, 0.0280, 0.1065),
                ("Effusion", 0.0191, 0.0050, 0.0326)]:
            row = ad[(ad["class"] == lbl) & (ad.arm_a == "A") & (ad.arm_b == "D")
                     & (ad.metric == "f1")].iloc[0]
            self.assertAlmostEqual(row.delta_point_estimate, exp_d, places=4)
            self.assertAlmostEqual(row.ci_lower, exp_lo, places=4)
            self.assertAlmostEqual(row.ci_upper, exp_hi, places=4)
            pe = [x for x in rep["primary_endpoints"] if x["label"] == lbl][0]
            self.assertEqual(pe["significance"], "significant")
            self.assertFalse(pe["bootstrap_covers_zero"])

    def test_threshold_stability_honest_and_vectorized(self):
        s = load_json(C.THRESHOLD_STABILITY_JSON)
        self.assertTrue(s["equivalence_check"]["passed"])
        self.assertEqual(s["per_label"]["Cardiomegaly"]["raw"]["f1_optimal"]
                         ["n_bootstraps"], 2000)
        cardio = s["per_label"]["Cardiomegaly"]
        self.assertAlmostEqual(cardio["raw"]["f1_optimal"]["first_fit"], 0.9021, places=4)
        self.assertFalse(cardio["raw"]["f1_optimal"]["within_pm05pct_of_first_fit"])
        for var in ("raw", "calibrated", "logistic"):
            for pol in ("youden", "sensitivity_constrained",
                        "precision_constrained"):
                self.assertIn("ci_width", cardio[var][pol])

    def test_ece_sensitivity_grid(self):
        e = load_json(C.ECE_SENSITIVITY_JSON)
        self.assertEqual(e["reference_binning"], {"n_bins": 15, "strategy": "equal_width"})
        self.assertEqual(len(e["rows"]), 36)
        ref = e["ece_at_reference_binning"]
        self.assertAlmostEqual(ref["Cardiomegaly"]["logistic"], 0.0029, places=4)
        self.assertAlmostEqual(ref["Effusion"]["logistic"], 0.0136, places=4)
        self.assertGreater(ref["Cardiomegaly"]["calibrated"],
                           ref["Cardiomegaly"]["raw"])

    def test_logistic_calibration_report(self):
        l = load_json(C.CALIBRATION_METRICS_DIR / "logistic_calibration_report.json")
        self.assertIn("LOGISTIC CALIBRATION FITTED ON VALIDATION ONLY", l["status"])
        self.assertGreater(l["parameters"]["Cardiomegaly"]["a"], 0)
        self.assertGreater(l["parameters"]["Effusion"]["a"], 0)
        # strictly monotone => AUROC/AUPRC invariant raw vs logistic
        inv = l["auroc_auprc_invariance_checks"]
        for k, v in inv.items():
            if k.endswith("_raw"):
                self.assertEqual(v, inv[k.replace("_raw", "_log")])
        self.assertLess(list(l["nll_after"].values())[0],
                        list(l["nll_before"].values())[0])

    def test_extension_arms_equalities_and_difference(self):
        e = load_json(C.EXT_EXTENSION_ARMS_JSON)
        metrics = ("accuracy", "precision", "recall", "specificity", "f1")
        for m in metrics:
            for lbl in C.TARGET_LABELS:
                self.assertEqual(e["per_label"][lbl]["A"][m],
                                 e["per_label"][lbl]["B"][m], f"{lbl} A!=B {m}")
                self.assertEqual(e["per_label"][lbl]["C"][m],
                                 e["per_label"][lbl]["F"][m], f"{lbl} C!=F {m}")
        card = e["per_label"]["Cardiomegaly"]
        self.assertAlmostEqual(card["E"]["f1"], 0.1376, places=4)
        self.assertAlmostEqual(card["E"]["precision"], 0.6400, places=4)
        self.assertAlmostEqual(card["E"]["recall"], 0.0771, places=4)

    def test_prevalence_shift_artifact(self):
        pr = load_json(C.PREVALENCE_SHIFT_JSON)
        self.assertEqual(pr["seed"], 42)
        card, eff = pr["per_label"]["Cardiomegaly"], pr["per_label"]["Effusion"]
        self.assertAlmostEqual(card["observed_prevalence"], 0.0261, places=4)
        self.assertAlmostEqual(eff["observed_prevalence"], 0.1257, places=4)
        cells = card["cohorts"] + eff["cohorts"]
        self.assertEqual(len(cells), 14)
        self.assertEqual(sum(1 for c in cells if c.get("simulated")), 8)
        self.assertEqual(sum(1 for c in cells if c.get("target_not_feasible")), 6)
        sim = [c for c in cells if c.get("simulated")]
        for c in sim:
            self.assertEqual(c["recall"], c["sensitivity"])
            self.assertFalse(c["target_not_feasible"])
        self.assertTrue(all(c["f1"] >= 0 for c in sim))

    def test_decision_policy_analysis_covers_extension_rows(self):
        df = pd.read_csv(C.DECISION_POLICY_ANALYSIS_CSV)
        self.assertEqual(len(df), 30)
        ext = load_json(C.EXT_EXTENSION_ARMS_JSON)
        f = ext["per_label"]["Cardiomegaly"]["F"]
        row = df[(df["class"] == "Cardiomegaly") & (df.variant == "logistic")
                 & (df.policy == "f1_optimal")].iloc[0]
        self.assertAlmostEqual(row["test_f1"], f["f1"], places=4)
        self.assertAlmostEqual(row["test_precision"], f["precision"], places=4)
        self.assertAlmostEqual(row["test_recall"], f["recall"], places=4)
    def test_snapshot_fields(self):
        snap = load_json(C.RESEARCH_SNAPSHOT_JSON)
        self.assertEqual(snap["baseline_id"], "BASELINE_v1")
        temperatures = snap["calibration"]["temperature"]
        self.assertAlmostEqual(temperatures["Cardiomegaly"], 1.1865, places=3)
        self.assertAlmostEqual(temperatures["Effusion"], 1.3979, places=3)
        self.assertEqual(snap["calibration"]["fitted_on"], "val")
        self.assertEqual(snap["thresholds"]["fitted_on"], ["val"])
        self.assertEqual(set(snap["thresholds"]["policies"]),
                         {"fixed", "f1_optimal", "youden",
                          "sensitivity_constrained", "precision_constrained"})
        self.assertFalse(snap["git"]["dirty"] is None)
        self.assertTrue(snap["checkpoints"]["best"]["exists"])


if __name__ == "__main__":
    unittest.main()