"""Auditoría A1–A12 del baseline congelado, sin abrir TEST."""

import csv
import json
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.preprocessing import AGGREGATE_FEATURES, CURRENT_CATEGORICAL_FEATURES, CURRENT_NUMERIC_FEATURES, load_split
from src.train_model_a import MODEL_A_FEATURES, approved_matrix


class ModelAAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = json.loads(Path("artefactos/model_a/model_a_metadata.json").read_text())
        cls.train = load_split("train")
        cls.validation = load_split("validation")
        cls.X_train = approved_matrix(cls.train)
        cls.X_validation = approved_matrix(cls.validation)
        cls.bundle = joblib.load("artefactos/model_a/model_a.joblib")

    def test_a1_exact_approved_schema(self) -> None:
        self.assertEqual(tuple(self.bundle["feature_names"]), MODEL_A_FEATURES)
        self.assertEqual(MODEL_A_FEATURES, (*CURRENT_NUMERIC_FEATURES, *CURRENT_CATEGORICAL_FEATURES, *AGGREGATE_FEATURES))

    def test_a2_no_leakage_columns(self) -> None:
        forbidden = ("fraud", "stage", "hard_negative", "target", "label", "transaction_id", "card_id")
        self.assertFalse(any(token in name.lower() for name in MODEL_A_FEATURES for token in forbidden))

    def test_a3_train_dimensions(self) -> None:
        self.assertEqual(len(self.X_train), len(self.train["y"]))

    def test_a4_validation_dimensions(self) -> None:
        self.assertEqual(len(self.X_validation), len(self.validation["y"]))

    def test_a5_example_ids_match_master_index(self) -> None:
        with Path("data/processed/example_index.csv").open(newline="") as handle:
            master = list(csv.DictReader(handle))
        self.assertTrue(all(master[i]["split"] == "train" for i in self.train["example_index"]))
        self.assertTrue(all(master[i]["split"] == "validation" for i in self.validation["example_index"]))

    def test_a6_risk_scores_in_unit_interval(self) -> None:
        for X in (self.X_train, self.X_validation):
            score = self.bundle["estimator"].predict_proba(X)[:, 1]
            self.assertTrue(np.all((score >= 0) & (score <= 1)))

    def test_a7_no_nan(self) -> None:
        self.assertFalse(np.isnan(self.X_train).any() or np.isnan(self.X_validation).any())

    def test_a8_no_infinity(self) -> None:
        self.assertTrue(np.isfinite(self.X_train).all() and np.isfinite(self.X_validation).all())

    def test_a9_fit_only_train(self) -> None:
        self.assertEqual(self.metadata["training_split"], "TRAIN")

    def test_a10_validation_only_for_selection(self) -> None:
        self.assertEqual(self.metadata["selection_split"], "VALIDATION")

    def test_a11_test_not_evaluated(self) -> None:
        self.assertFalse(self.metadata["test_evaluated"])
        self.assertFalse(Path("artefactos/model_a/test_scores.csv").exists())

    def test_a12_reproducible(self) -> None:
        self.assertLessEqual(self.metadata["reproducibility_absolute_difference"], 1e-12)


if __name__ == "__main__":
    unittest.main()
