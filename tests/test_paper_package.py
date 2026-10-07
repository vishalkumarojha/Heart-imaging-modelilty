"""Tests for the paper package (final phase).

Verifies the deliverables under `outputs/paper/`: the 9 tables, 9 figures,
evidence map, claim audit, reproducibility manifest, prose documents, and the
number-hygiene discipline ("no paper number without a traceable source").

Run:  python -m unittest discover -s tests -v
"""
import json
import unittest
from pathlib import Path

import pandas as pd

from src import config as C
from src.paper_build import check_prose_numbers

TABLES = {
    "table_01_dataset.csv": 4,
    "table_02_baseline.csv": 10,
    "table_03_main_results.csv": 6,
    "table_04_ablation.csv": 8,
    "table_05_thresholds.csv": 10,
    "table_06_calibration.csv": 10,
    "table_07_error_analysis.csv": 5,
    "table_08_explainability.csv": 10,
    "table_09_external_validation_status.csv": 5,
}

FIGURES = [
    "figure_01_architecture.png", "figure_02_dataset_distribution.png",
    "figure_03_roc.png", "figure_04_precision_recall.png",
    "figure_05_calibration.png", "figure_06_threshold_analysis.png",
    "figure_07_confusion_matrix.png", "figure_08_error_analysis.png",
    "figure_09_gradcam.png",
]

DOCS = [
    "research_question.md", "contribution_statement.md", "paper_outline.md",
    "draft_results.md", "draft_discussion.md", "draft_limitations.md",
    "titles.md", "abstract.md", "conclusion.md",
]


class TestPaperTables(unittest.TestCase):
    def test_all_nine_tables_exist_with_expected_row_counts(self):
        for name, rows in TABLES.items():
            p = C.PAPER_TABLES_DIR / name
            self.assertTrue(p.exists(), f"missing table {name}")
            df = pd.read_csv(p)
            self.assertEqual(len(df), rows, name)

    def test_table_main_results_values_match_master(self):
        m = pd.read_csv(C.MASTER_RESULTS_CSV)
        t = pd.read_csv(C.PAPER_TABLES_DIR / "table_03_main_results.csv")
        a = t[t.arm.str.startswith("A (baseline")].iloc[0]
        d = t[t.arm.str.startswith("D (calibration")].iloc[0]
        ma = m[(m.experiment_id == "A") & (m["class"] == "Cardiomegaly")]
        md = m[(m.experiment_id == "D") & (m["class"] == "Cardiomegaly")
               & (m.threshold_policy == "f1_optimal")]
        self.assertAlmostEqual(a["f1"], ma.f1.iloc[0], places=4)
        self.assertAlmostEqual(d["f1"], md.f1.iloc[0], places=4)

    def test_table_09_is_honest_and_pending(self):
        df = pd.read_csv(C.PAPER_TABLES_DIR / "table_09_external_validation_status.csv")
        status = df[df.field == "status"].value.iloc[0]
        self.assertEqual(status, "EXTERNAL VALIDATION PENDING — DATASET UNAVAILABLE")
        self.assertEqual(df[df.field == "no_metrics_were_computed"].value.iloc[0],
                         "true")


class TestPaperFigures(unittest.TestCase):
    def test_all_nine_figures_exist_and_are_nonempty(self):
        for name in FIGURES:
            p = C.PAPER_FIGURES_DIR / name
            self.assertTrue(p.exists(), f"missing figure {name}")
            self.assertGreater(p.stat().st_size, 5000, name)

    def test_no_external_validation_figure(self):
        for name in C.PAPER_FIGURES_DIR.glob("*.png"):
            self.assertNotIn("external", name.name.lower())


class TestEvidenceMap(unittest.TestCase):
    def test_every_claim_verified(self):
        em = json.loads(C.PAPER_EVIDENCE_MAP_JSON.read_text())
        self.assertGreaterEqual(em["n_claims"], 60)
        self.assertEqual(em["n_verified"], em["n_claims"])

    def test_every_source_artifact_exists(self):
        em = json.loads(C.PAPER_EVIDENCE_MAP_JSON.read_text())
        for c in em["claims"]:
            src = c["source_artifact"]
            if src == "tests":
                self.assertTrue((C.PROJECT_ROOT / "tests").exists())
            else:
                self.assertTrue((C.PROJECT_ROOT / src).exists(),
                                f"missing source {src} for claim: {c['claim']}")


class TestClaimAudit(unittest.TestCase):
    def test_audit_columns_and_verification(self):
        df = pd.read_csv(C.PAPER_CLAIM_AUDIT_CSV)
        self.assertEqual(list(df.columns),
                         ["claim", "section", "numeric_or_text",
                          "source_artifact", "verified", "notes"])
        self.assertTrue((df.verified == True).all(), msg=df[df.verified != True])  # noqa: E712
        self.assertEqual(len(df), json.loads(
            C.PAPER_EVIDENCE_MAP_JSON.read_text())["n_claims"])


class TestReproducibilityManifest(unittest.TestCase):
    def test_manifest_supplies_required_fields(self):
        m = json.loads(C.PAPER_REPRO_MANIFEST_JSON.read_text())
        for key in ("model", "checkpoint", "checkpoint_sha256", "dataset",
                    "split", "seed", "preprocessing", "training_configuration",
                    "temperatures", "thresholds", "package_environment",
                    "test_count_collected", "git", "references"):
            self.assertIn(key, m)
        self.assertAlmostEqual(m["temperatures"]["Cardiomegaly"], 1.1865, places=4)
        self.assertIn("research_snapshot.json", m["references"]["research_snapshot"])
        self.assertIsInstance(m["test_count_collected"], int)

    def test_manifest_matches_snapshot_checkpoint(self):
        m = json.loads(C.PAPER_REPRO_MANIFEST_JSON.read_text())
        snap = json.loads(C.RESEARCH_SNAPSHOT_JSON.read_text())
        self.assertEqual(m["checkpoint_sha256"],
                         snap["checkpoints"]["best"]["sha256"])


class TestPaperDocs(unittest.TestCase):
    def test_all_prose_documents_exist(self):
        for name in DOCS:
            self.assertTrue((C.PAPER_DIR / name).exists(), name)

    def test_external_validation_sentence_is_exact(self):
        text = " ".join((C.PAPER_DIR / "draft_results.md").read_text().split())
        self.assertIn("External validation was not completed because the "
                      "required external cohort was unavailable.", text)
        self.assertNotIn("external AUROC", text)

    def test_abstract_reports_exact_measured_numbers(self):
        text = (C.PAPER_DIR / "abstract.md").read_text()
        self.assertIn("0.8972", text)
        self.assertIn("0.3536", text)
        self.assertIn("0.4866", text)


class TestNumberHygiene(unittest.TestCase):
    def test_every_prose_number_has_a_traceable_source(self):
        violations = check_prose_numbers()
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()