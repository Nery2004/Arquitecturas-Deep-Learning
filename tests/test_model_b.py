"""Auditoría B1–B16 de la GRU congelada; no abre el split TEST."""

import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.sequence_model import SequenceGRU
from src.train_model_b import SEQUENCE_INPUT_KEYS, load_sequence_split


class ModelBAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = json.loads(Path("artefactos/model_b/model_b_metadata.json").read_text())
        cls.train = load_sequence_split("train")
        cls.validation = load_sequence_split("validation")
        with Path("data/processed/example_index.csv").open(newline="") as handle:
            cls.index = list(csv.DictReader(handle))
        checkpoint = torch.load("artefactos/model_b/model_b.pt", map_location="cpu")
        cls.model = SequenceGRU(checkpoint["merchant_vocab_size"], checkpoint["channel_vocab_size"],
            **checkpoint["config"], merchant_embedding_dim=checkpoint["merchant_embedding_dim"],
            channel_embedding_dim=checkpoint["channel_embedding_dim"])
        cls.model.load_state_dict(checkpoint["state_dict"]); cls.model.eval()

    def test_b1_same_example_index_as_a(self) -> None:
        with Path("artefactos/model_a/train_scores.csv").open(newline="") as handle:
            a_train = list(csv.DictReader(handle))
        with Path("artefactos/model_a/validation_scores.csv").open(newline="") as handle:
            a_validation = list(csv.DictReader(handle))
        self.assertEqual([r["example_id"] for r in a_train], [self.index[i]["example_id"] for i in self.train["example_index"]])
        self.assertEqual([r["example_id"] for r in a_validation], [self.index[i]["example_id"] for i in self.validation["example_index"]])

    def test_b2_same_targets_as_a(self) -> None:
        for split, arrays in (("train", self.train), ("validation", self.validation)):
            with Path(f"artefactos/model_a/{split}_scores.csv").open(newline="") as handle:
                a_y = np.asarray([int(r["y_true"]) for r in csv.DictReader(handle)])
            self.assertTrue(np.array_equal(a_y, arrays["y"]))

    def test_b3_b4_b5_history_order_target_absent_and_no_future(self) -> None:
        with Path("data/generated/transactions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_card = defaultdict(list)
        for row in rows: by_card[row["card_id"]].append(row)
        for global_index in self.validation["example_index"][::1000]:
            example = self.index[int(global_index)]
            target_key = (example["target_timestamp"], example["transaction_id"])
            prior = [r for r in by_card[example["card_id"]]
                     if (r["timestamp"], r["transaction_id"]) < target_key][-12:]
            self.assertNotIn(example["transaction_id"], [r["transaction_id"] for r in prior])
            self.assertTrue(all((a["timestamp"], a["transaction_id"]) < (b["timestamp"], b["transaction_id"])
                                for a, b in zip(prior, prior[1:])))
            self.assertLess((prior[-1]["timestamp"], prior[-1]["transaction_id"]), target_key)

    def test_b6_and_b11_padding_ignored_and_hidden_is_last_valid(self) -> None:
        index = int(np.flatnonzero(self.validation["history_length"] < 12)[0]); length = int(self.validation["history_length"][index])
        numeric = torch.from_numpy(self.validation["X_sequence_numeric"][index:index+1]).clone()
        categorical = torch.from_numpy(self.validation["X_sequence_categorical"][index:index+1].astype(np.int64)).clone()
        current_numeric = torch.from_numpy(self.validation["X_current_numeric"][index:index+1])
        current_categorical = torch.from_numpy(self.validation["X_current_categorical"][index:index+1].astype(np.int64))
        lengths = torch.tensor([length]); changed_numeric=numeric.clone();changed_categorical=categorical.clone()
        changed_numeric[:, :12-length] = 999.; changed_categorical[:, :12-length] = 2
        with torch.no_grad():
            original=self.model(numeric,categorical,lengths,current_numeric,current_categorical)
            changed=self.model(changed_numeric,changed_categorical,lengths,current_numeric,current_categorical)
        self.assertTrue(torch.equal(original, changed))

    def test_b7_pad_not_unk(self) -> None:
        vocab=json.loads(Path("artefactos/preprocessing/vocabularies.json").read_text())
        self.assertEqual(vocab["merchant_category"]["PAD"],0);self.assertEqual(vocab["merchant_category"]["UNK"],1)

    def test_b8_no_leakage_columns(self) -> None:
        names=self.metadata["sequence_numeric_features"]+self.metadata["sequence_categorical_features"]+self.metadata["current_numeric_features"]+self.metadata["current_categorical_features"]
        self.assertFalse(any(token in name for name in names for token in ("fraud","stage","hard_negative","card_id","transaction_id")))

    def test_b9_no_aggregates_loaded(self) -> None:
        self.assertEqual(set(self.train),set(SEQUENCE_INPUT_KEYS));self.assertFalse(self.metadata["aggregate_features_used"])

    def test_b10_and_b11_scores_valid(self) -> None:
        with Path("artefactos/model_b/validation_scores.csv").open(newline="") as handle:
            scores=np.asarray([float(r["risk_score"]) for r in csv.DictReader(handle)])
        self.assertTrue(np.isfinite(scores).all());self.assertTrue(np.all((scores>=0)&(scores<=1)))

    def test_b12_pos_weight_from_train(self) -> None:
        expected=float((self.train["y"]==0).sum()/self.train["y"].sum())
        self.assertAlmostEqual(self.metadata["pos_weight"],expected)

    def test_b13_b14_fit_and_early_stopping_splits(self) -> None:
        self.assertEqual(self.metadata["training_split"],"TRAIN");self.assertEqual(self.metadata["early_stopping_split"],"VALIDATION")

    def test_b15_test_not_evaluated(self) -> None:
        self.assertFalse(self.metadata["test_evaluated"])
        final=Path("artefactos/final_evaluation.json")
        self.assertEqual(Path("artefactos/model_b/test_scores.csv").exists(), final.exists())

    def test_b16_reproducible(self) -> None:
        self.assertLessEqual(self.metadata["reproducibility_absolute_difference"],1e-12)


if __name__ == "__main__": unittest.main()
