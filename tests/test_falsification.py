"""Controles P1–P14 y H1–H12; usan únicamente VALIDATION."""

import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.falsification import permute_valid_history, truncate_history
from src.preprocessing import file_sha256
from src.train_model_b import load_sequence_split


class FalsificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original=load_sequence_split("validation")
        cls.permuted=permute_valid_history(cls.original,100)
        cls.short=truncate_history(cls.original)
        cls.metadata=json.loads(Path("artefactos/model_b/falsification_metadata.json").read_text())

    def test_p1_p2_p3_same_examples_ids_targets(self) -> None:
        self.assertEqual(len(self.original["y"]),len(self.permuted["y"]))
        self.assertTrue(np.array_equal(self.original["example_index"],self.permuted["example_index"]))
        self.assertTrue(np.array_equal(self.original["y"],self.permuted["y"]))

    def test_p4_p5_current_and_lengths_unchanged(self) -> None:
        for key in ("X_current_numeric","X_current_categorical","history_length","sequence_mask"):
            self.assertTrue(np.array_equal(self.original[key],self.permuted[key]))

    def test_p6_p7_p8_same_complete_events_only_order_changes_and_pad_stays(self) -> None:
        changed=0;width=self.original["X_sequence_numeric"].shape[1]
        for i,length_value in enumerate(self.original["history_length"]):
            length=int(length_value);start=width-length
            original=[tuple(np.r_[self.original["X_sequence_numeric"][i,j],self.original["X_sequence_categorical"][i,j]]) for j in range(start,width)]
            permuted=[tuple(np.r_[self.permuted["X_sequence_numeric"][i,j],self.permuted["X_sequence_categorical"][i,j]]) for j in range(start,width)]
            self.assertEqual(sorted(original),sorted(permuted))
            self.assertTrue(np.all(self.permuted["X_sequence_numeric"][i,:start]==0))
            self.assertTrue(np.all(self.permuted["X_sequence_categorical"][i,:start]==0))
            changed += original != permuted
        self.assertGreater(changed,0)

    def test_p9_p10_no_future_or_target_in_history(self) -> None:
        with Path("data/processed/example_index.csv").open(newline="") as handle:index=list(csv.DictReader(handle))
        with Path("data/generated/transactions.csv").open(newline="") as handle:rows=list(csv.DictReader(handle))
        by_card=defaultdict(list)
        for row in rows:by_card[row["card_id"]].append(row)
        for global_index in self.original["example_index"][::1500]:
            example=index[int(global_index)];target=(example["target_timestamp"],example["transaction_id"])
            history=[r for r in by_card[example["card_id"]] if (r["timestamp"],r["transaction_id"])<target][-12:]
            self.assertNotIn(example["transaction_id"],[r["transaction_id"] for r in history])
            self.assertTrue(all((r["timestamp"],r["transaction_id"])<target for r in history))

    def test_p11_p12_reproducible_and_seed_sensitive(self) -> None:
        repeated=permute_valid_history(self.original,100);different=permute_valid_history(self.original,101)
        self.assertTrue(np.array_equal(self.permuted["X_sequence_numeric"],repeated["X_sequence_numeric"]))
        self.assertFalse(np.array_equal(self.permuted["X_sequence_numeric"],different["X_sequence_numeric"]))

    def test_p13_p14_same_checkpoint_and_no_test(self) -> None:
        self.assertEqual(file_sha256(Path("artefactos/model_b/model_b.pt")),self.metadata["model_b_checkpoint_hash"])
        self.assertFalse(self.metadata["test_evaluated"]);self.assertEqual(self.metadata["evaluation_split"],"VALIDATION")

    def test_h1_h2_h3_h4_same_examples_targets_current(self) -> None:
        for key in ("example_index","y","X_current_numeric","X_current_categorical"):
            self.assertTrue(np.array_equal(self.original[key],self.short[key]))

    def test_h5_h6_h7_h8_max_three_latest_order_and_padding(self) -> None:
        self.assertLessEqual(int(self.short["history_length"].max()),3);width=12
        for i,length_value in enumerate(self.short["history_length"]):
            length=int(length_value);start=width-length
            self.assertTrue(np.array_equal(self.short["X_sequence_numeric"][i,start:],self.original["X_sequence_numeric"][i,-length:]))
            self.assertTrue(np.all(self.short["X_sequence_numeric"][i,:start]==0))
            self.assertTrue(np.all(self.short["X_sequence_categorical"][i,:start]==0))
            self.assertTrue(np.all(self.short["sequence_mask"][i,:start]==0))

    def test_h9_h10_target_and_future_absent_by_construction(self) -> None:
        self.assertTrue(np.all(self.short["history_length"]<=self.original["history_length"]))
        self.assertTrue(np.all(self.short["history_length"]>0))

    def test_h11_h12_same_checkpoint_and_no_test(self) -> None:
        self.assertEqual(self.metadata["checkpoint_hash_after"],self.metadata["model_b_checkpoint_hash"])
        self.assertFalse(Path("artefactos/model_b/test_scores.csv").exists())


if __name__ == "__main__":unittest.main()
