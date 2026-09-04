"""Controles de los artefactos procesados ya generados."""

import csv
import json
import unittest
from pathlib import Path

import numpy as np

from src.preprocessing import PAD_INDEX, UNK_INDEX, file_sha256, load_split, load_split_config


class ProcessedDataTests(unittest.TestCase):
    def test_source_fingerprint_and_frozen_split(self) -> None:
        source = Path("data/generated/transactions.csv")
        metadata = json.loads(Path("data/generated/dataset_metadata.json").read_text())
        split = load_split_config()
        self.assertEqual(file_sha256(source), metadata["dataset_fingerprint"])
        self.assertEqual(split["dataset_fingerprint"], metadata["dataset_fingerprint"])

    def test_models_share_index_and_target(self) -> None:
        total = 0
        with Path("data/processed/example_index.csv").open(newline="") as handle:
            index_rows = list(csv.DictReader(handle))
        for split in ("train", "validation", "test"):
            arrays = load_split(split)
            indices = arrays["example_index"]
            self.assertTrue(np.array_equal(arrays["y"], np.asarray([int(index_rows[i]["target"]) for i in indices])))
            self.assertTrue(all(index_rows[i]["split"] == split for i in indices))
            total += len(indices)
        self.assertEqual(total, len(index_rows))

    def test_padding_mask_and_categories(self) -> None:
        for split in ("train", "validation", "test"):
            arrays = load_split(split)
            mask = arrays["sequence_mask"]
            categories = arrays["X_sequence_categorical"]
            self.assertTrue(np.all(categories[~mask] == PAD_INDEX))
            self.assertTrue(np.all(categories[mask] >= 2))
            self.assertTrue(np.array_equal(mask.sum(axis=1), arrays["history_length"]))
        self.assertNotEqual(PAD_INDEX, UNK_INDEX)

    def test_preprocessors_report_train_fit(self) -> None:
        for name in ("aggregate_scaler", "current_scaler", "sequence_scaler"):
            artifact = json.loads(Path(f"artefactos/preprocessing/{name}.json").read_text())
            self.assertEqual(artifact["fitted_split"], "train")


if __name__ == "__main__":
    unittest.main()
