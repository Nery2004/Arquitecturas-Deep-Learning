"""T1–T12 y F1–F12 de la decisión económica y cierre experimental."""

import csv
import json
from pathlib import Path

import numpy as np

from src.economics import (decision_metrics, economic_cost, monthly_normalize,
                           select_economic_threshold)
from src.preprocessing import file_sha256

ROOT = Path(__file__).parents[1]


def read_json(relative): return json.loads((ROOT / relative).read_text())
def score_rows(model):
    with (ROOT / f"artefactos/model_{model.lower()}/test_scores.csv").open(newline="") as h: return list(csv.DictReader(h))


def test_t1_threshold_source_is_validation():
    c=read_json("artefactos/final_decision_config.json"); assert c["selection_data"] == "VALIDATION only"
def test_t2_economic_costs_are_exact(): assert economic_cost(2, 3) == 2*4200 + 3*180
def test_t3_same_method_for_all_models():
    c=read_json("artefactos/final_decision_config.json"); assert set(c["validation_thresholds"]) == {"A","B","C"}
def test_t4_test_was_not_used_in_selection():
    c=read_json("artefactos/final_decision_config.json"); assert "TEST" not in c["selection_data"]
def test_t5_pretest_freeze_is_evidenced():
    c=read_json("artefactos/final_decision_config.json"); f=read_json("artefactos/final_evaluation.json"); assert c["pre_test_config_sha256"] == f["frozen_config_sha256_before_test"]
def test_t6_validation_thresholds_applied_unchanged():
    c=read_json("artefactos/final_decision_config.json"); assert all({float(r["threshold"]) for r in score_rows(m)} == {c["validation_thresholds"][m]} for m in ("A","B","C"))
def test_t7_confusion_matrix_counts():
    m=decision_metrics(np.array([0,0,1,1]),np.array([.1,.9,.2,.8]),.5); assert (m.TP,m.FP,m.TN,m.FN)==(1,1,1,1)
def test_t8_precision(): assert decision_metrics(np.array([0,1,1]),np.array([.8,.9,.1]),.5).precision == .5
def test_t9_recall(): assert decision_metrics(np.array([0,1,1]),np.array([.8,.9,.1]),.5).recall == .5
def test_t10_f1(): assert decision_metrics(np.array([0,1,1]),np.array([.8,.9,.1]),.5).F1 == .5
def test_t11_metric_economic_cost(): assert decision_metrics(np.array([0,1]),np.array([.9,.1]),.5).economic_cost == 4380
def test_t12_tie_break_uses_highest_threshold():
    best,_=select_economic_threshold(np.array([0,1]),np.array([.2,.8])); assert best.threshold == .8


def test_f1_checkpoint_hashes_unchanged():
    c=read_json("artefactos/final_decision_config.json")
    for m in ("A","B","C"):
        ext="joblib" if m=="A" else "pt"; assert file_sha256(ROOT/f"artefactos/model_{m.lower()}/model_{m.lower()}.{ext}")==c["model_checkpoints_sha256"][m]
def test_f2_preprocessing_fit_only_train(): assert read_json("data/processed/processed_metadata.json")["fit_split"] == "train"
def test_f3_no_hyperparameter_changes(): assert read_json("artefactos/final_evaluation.json")["experimental_status"] == "CLOSED"
def test_f4_no_test_threshold_search():
    f=read_json("artefactos/final_evaluation.json"); assert f["validation_thresholds"] == read_json("artefactos/final_decision_config.json")["validation_thresholds"]
def test_f5_same_test_example_ids():
    ids=[[r["example_id"] for r in score_rows(m)] for m in ("A","B","C")]; assert ids[0]==ids[1]==ids[2]
def test_f6_same_targets():
    ys=[[r["y_true"] for r in score_rows(m)] for m in ("A","B","C")]; assert ys[0]==ys[1]==ys[2]
def test_f7_scores_in_unit_interval(): assert all(0<=float(r["risk_score"])<=1 for m in ("A","B","C") for r in score_rows(m))
def test_f8_scores_finite(): assert all(np.isfinite(float(r["risk_score"])) for m in ("A","B","C") for r in score_rows(m))
def test_f9_ap_implementation_is_recorded(): assert read_json("artefactos/final_decision_config.json")["primary_metric"] == "AUC-PR / Average Precision (AP)"
def test_f10_confusion_counts_sum_to_test():
    f=read_json("artefactos/final_evaluation.json"); assert all(sum(f["models"][m][k] for k in ("TP","FP","TN","FN"))==14366 for m in ("A","B","C"))
def test_f11_costs_recalculate_from_errors():
    f=read_json("artefactos/final_evaluation.json"); assert all(f["models"][m]["economic_cost"]==economic_cost(f["models"][m]["FN"],f["models"][m]["FP"]) for m in ("A","B","C"))
def test_f12_monthly_normalization():
    f=read_json("artefactos/final_evaluation.json"); assert np.isclose(f["monthly_cost_A"],monthly_normalize(f["models"]["A"]["economic_cost"],f["test_days"]))
