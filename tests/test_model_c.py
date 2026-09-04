"""Auditoría C1–C20 del híbrido congelado, sin cargar TEST."""

import csv
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from src.hybrid_model import HybridGRU
from src.preprocessing import AGGREGATE_FEATURES, file_sha256
from src.train_model_c import C_INPUT_KEYS, load_hybrid_split


class ModelCAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls)->None:
        cls.meta=json.loads(Path("artefactos/model_c/model_c_metadata.json").read_text())
        cls.train=load_hybrid_split("train");cls.val=load_hybrid_split("validation")
        with Path("data/processed/example_index.csv").open(newline="") as h:cls.index=list(csv.DictReader(h))
        checkpoint=torch.load("artefactos/model_c/model_c.pt",map_location="cpu")
        config=checkpoint["config"]
        cls.model=HybridGRU(checkpoint["merchant_vocab_size"],checkpoint["channel_vocab_size"],
            aggregate_size=checkpoint["aggregate_size"],aggregate_hidden=config["aggregate_hidden"],
            fusion_hidden=config["fusion_hidden"],dropout=config["dropout"])
        cls.model.load_state_dict(checkpoint["state_dict"]);cls.model.eval()

    def test_c1_c2_c3_same_universe_targets_splits(self)->None:
        for split,arrays in (("train",self.train),("validation",self.val)):
            for prior in ("model_a","model_b"):
                with Path(f"artefactos/{prior}/{split}_scores.csv").open(newline="") as h:rows=list(csv.DictReader(h))
                self.assertEqual([r["example_id"] for r in rows],[self.index[i]["example_id"] for i in arrays["example_index"]])
                self.assertTrue(np.array_equal(np.asarray([int(r["y_true"]) for r in rows]),arrays["y"]))

    def test_c4_c5_sequence_and_current_match_b(self)->None:
        b=json.loads(Path("artefactos/model_b/model_b_metadata.json").read_text())
        self.assertEqual(self.meta["gru"]["hidden_size"],b["hidden_size"]);self.assertEqual(self.meta["gru"]["num_layers"],b["num_layers"])
        self.assertEqual(self.meta["embeddings"]["merchant"],b["merchant_embedding_dim"]);self.assertEqual(self.meta["current_branch_hidden"],b["dense_size"])

    def test_c6_c7_frozen_causal_aggregates(self)->None:
        self.assertEqual(tuple(self.meta["aggregate_branch"]["input_features"]),AGGREGATE_FEATURES)
        processed=json.loads(Path("data/processed/processed_metadata.json").read_text())
        self.assertTrue(processed["validation"]["history_only_before_target"])

    def test_c8_no_exact_current_duplication(self)->None:
        current=set(self.meta["current_numeric_features"]+self.meta["current_categorical_features"])
        self.assertFalse(current & set(self.meta["aggregate_branch"]["input_features"]))

    def test_c9_c10_c11_no_metadata_or_ids(self)->None:
        names=self.meta["sequence_numeric_features"] if "sequence_numeric_features" in self.meta else []
        names+=self.meta["aggregate_branch"]["input_features"]
        self.assertFalse(any(token in name for name in names for token in ("fraud","stage","hard_negative","card_id","transaction_id")))

    def test_c12_padding_ignored(self)->None:
        i=0;length=int(self.val["history_length"][i]);batch=lambda key:torch.from_numpy(self.val[key][i:i+1])
        numeric=batch("X_sequence_numeric").clone();categorical=batch("X_sequence_categorical").long().clone();changed_n=numeric.clone();changed_c=categorical.clone();changed_n[:,:12-length]=999;changed_c[:,:12-length]=2
        args=(torch.tensor([length]),batch("X_current_numeric"),batch("X_current_categorical").long(),batch("X_aggregate"))
        with torch.no_grad():self.assertTrue(torch.equal(self.model(numeric,categorical,*args),self.model(changed_n,changed_c,*args)))

    def test_c13_c14_scores_valid(self)->None:
        with Path("artefactos/model_c/validation_scores.csv").open(newline="") as h:s=np.asarray([float(r["risk_score"]) for r in csv.DictReader(h)])
        self.assertTrue(np.isfinite(s).all());self.assertTrue(np.all((s>=0)&(s<=1)))

    def test_c15_pos_weight_train_only(self)->None:
        self.assertAlmostEqual(self.meta["pos_weight"],float((self.train["y"]==0).sum()/self.train["y"].sum()))

    def test_c16_c17_fit_and_early_stopping(self)->None:
        self.assertEqual(self.meta["training_split"],"TRAIN");self.assertEqual(self.meta["early_stopping_split"],"VALIDATION")

    def test_c18_test_not_evaluated(self)->None:
        self.assertFalse(self.meta["test_evaluated"])
        final=Path("artefactos/final_evaluation.json")
        self.assertEqual(Path("artefactos/model_c/test_scores.csv").exists(),final.exists())

    def test_c19_ablation_same_checkpoint(self)->None:
        self.assertEqual(file_sha256(Path("artefactos/model_c/model_c.pt")),self.meta["model_fingerprint_sha256"])
        self.assertGreater(self.meta["ablation_normal_ap"],self.meta["ablation_aggregates_neutral_ap"])

    def test_c20_reproducible(self)->None:
        self.assertLessEqual(self.meta["reproducibility_absolute_difference"],1e-12)


if __name__=="__main__":unittest.main()
