"""Entrena y congela Modelo A usando TRAIN y seleccionando con VALIDATION.

TEST no se carga ni se predice. Ejecución: ``python3 -m src.train_model_a``.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import RANDOM_SEED
from .evaluation import PRIMARY_METRIC_NAME, average_precision, pr_curve
from .models import create_model_a
from .preprocessing import (AGGREGATE_FEATURES, CURRENT_CATEGORICAL_FEATURES,
                            CURRENT_NUMERIC_FEATURES, file_sha256, load_split,
                            processed_fingerprint)

MODEL_A_FEATURES = (*CURRENT_NUMERIC_FEATURES, *CURRENT_CATEGORICAL_FEATURES, *AGGREGATE_FEATURES)
CATEGORICAL_INDICES = [8, 9]
SUSPICIOUS_TOKENS = ("fraud", "stage", "attack", "hard_negative", "target", "label",
                     "transaction_id", "card_id")


def approved_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Combina solamente current y agregados; nunca toca arrays secuenciales."""
    return np.column_stack((arrays["X_current_numeric"], arrays["X_current_categorical"],
                            arrays["X_aggregate"])).astype(np.float64)


def verify_fingerprints(root: Path) -> tuple[dict[str, Any], str]:
    source_meta = json.loads((root / "data/generated/dataset_metadata.json").read_text())
    assert file_sha256(root / "data/generated/transactions.csv") == source_meta["dataset_fingerprint"]
    processed_meta = json.loads((root / "data/processed/processed_metadata.json").read_text())
    artifact_dir = root / "artefactos/preprocessing"
    paths = [root / "data/processed/split_config.json", root / "data/processed/example_index.csv",
             root / "data/processed/aggregate_features_raw.csv",
             *(artifact_dir / f"{name}.json" for name in
               ("aggregate_scaler", "current_scaler", "sequence_scaler", "vocabularies")),
             *(root / f"data/processed/model_inputs_{split}.npz" for split in ("train", "validation", "test"))]
    assert processed_fingerprint(paths) == processed_meta["processed_fingerprint"]
    split_hash = file_sha256(root / "data/processed/split_config.json")
    return processed_meta, split_hash


def load_analysis_index(root: Path) -> list[dict[str, str]]:
    with (root / "data/processed/example_index.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def explicit_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Pesos balanceados calculados exclusivamente desde y_train."""
    n = len(y_train); positives = int(y_train.sum()); negatives = n - positives
    return {0: n / (2 * negatives), 1: n / (2 * positives)}


def candidate_definitions(weights: dict[int, float]) -> list[dict[str, Any]]:
    return [
        {"candidate_id": "logistic_sanity", "algorithm": "LogisticRegression",
         "parameters": {"C": 1.0, "max_iter": 500, "solver": "lbfgs", "class_weight": weights,
                        "random_state": RANDOM_SEED}},
        {"candidate_id": "hgb_01", "algorithm": "HistGradientBoostingClassifier",
         "parameters": {"learning_rate": .05, "max_iter": 150, "max_leaf_nodes": 15,
                        "min_samples_leaf": 50, "l2_regularization": 1., "random_state": RANDOM_SEED}},
        {"candidate_id": "hgb_02", "algorithm": "HistGradientBoostingClassifier",
         "parameters": {"learning_rate": .08, "max_iter": 180, "max_leaf_nodes": 31,
                        "min_samples_leaf": 40, "l2_regularization": 1., "random_state": RANDOM_SEED}},
        {"candidate_id": "hgb_03", "algorithm": "HistGradientBoostingClassifier",
         "parameters": {"learning_rate": .05, "max_iter": 220, "max_leaf_nodes": 31,
                        "min_samples_leaf": 80, "l2_regularization": 2., "random_state": RANDOM_SEED}},
        {"candidate_id": "hgb_04", "algorithm": "HistGradientBoostingClassifier",
         "parameters": {"learning_rate": .10, "max_iter": 150, "max_leaf_nodes": 15,
                        "min_samples_leaf": 80, "l2_regularization": 2., "random_state": RANDOM_SEED}},
        {"candidate_id": "hgb_05", "algorithm": "HistGradientBoostingClassifier",
         "parameters": {"learning_rate": .05, "max_iter": 180, "max_leaf_nodes": 63,
                        "min_samples_leaf": 80, "l2_regularization": 5., "random_state": RANDOM_SEED}},
    ]


def build_candidate(definition: dict[str, Any]):
    if definition["algorithm"] == "LogisticRegression":
        transform = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_INDICES)],
                                      remainder="passthrough")
        return Pipeline([("one_hot", transform), ("model", LogisticRegression(**definition["parameters"]))])
    return create_model_a(definition["parameters"])


def train_candidates(X_train: np.ndarray, y_train: np.ndarray, X_validation: np.ndarray,
                     y_validation: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results, fitted = [], {}
    weights = explicit_class_weights(y_train)
    sample_weight = np.where(y_train == 1, weights[1], weights[0])
    for definition in candidate_definitions(weights):
        model = build_candidate(definition)
        started = time.perf_counter()
        if definition["algorithm"] == "LogisticRegression":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        elapsed = time.perf_counter() - started
        train_score = model.predict_proba(X_train)[:, 1]
        validation_score = model.predict_proba(X_validation)[:, 1]
        row = {**definition, "train_ap": average_precision(y_train, train_score),
               "validation_ap": average_precision(y_validation, validation_score),
               "train_validation_gap": average_precision(y_train, train_score) -
                                       average_precision(y_validation, validation_score),
               "fit_time_seconds": elapsed}
        results.append(row); fitted[definition["candidate_id"]] = model
    best_ap = max(r["validation_ap"] for r in results)
    # Dentro de 0.001 se elige el primero de la lista, que ordena de simple a complejo.
    selected_row = next(r for r in results if r["validation_ap"] >= best_ap - .001)
    return results, {"result": selected_row, "model": fitted[selected_row["candidate_id"]]}


def write_scores(path: Path, indices: np.ndarray, index_rows: list[dict[str, str]], y: np.ndarray,
                 scores: np.ndarray, split: str) -> None:
    with path.open("w", newline="") as handle:
        fields = ("example_id", "transaction_id", "target_timestamp", "y_true", "risk_score", "split",
                  "fraud_type", "hard_negative_type")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for local, global_index in enumerate(indices):
            row = index_rows[int(global_index)]
            writer.writerow({"example_id": row["example_id"], "transaction_id": row["transaction_id"],
                "target_timestamp": row["target_timestamp"], "y_true": int(y[local]),
                "risk_score": f"{scores[local]:.17g}", "split": split,
                "fraud_type": row["fraud_type"], "hard_negative_type": row["hard_negative_type"]})


def save_pr_figure(y_validation: np.ndarray, scores: np.ndarray, path: Path) -> None:
    precision, recall, _ = pr_curve(y_validation, scores)
    prevalence = float(y_validation.mean())
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.plot(recall, precision, label="Modelo A")
    axis.axhline(prevalence, linestyle="--", color="gray", label=f"Prevalencia ({prevalence:.3%})")
    axis.set(title="Precision–Recall — Modelo A — Validation", xlabel="Recall", ylabel="Precision",
             xlim=(0, 1), ylim=(0, 1)); axis.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def score_group_summary(index_rows: list[dict[str, str]], indices: np.ndarray,
                        scores: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    fraud_groups: dict[str, list[float]] = {name: [] for name in
        ("testing_cashout", "channel_takeover", "amount_anomaly")}
    hard_groups: dict[str, list[float]] = {}
    for local, global_index in enumerate(indices):
        row = index_rows[int(global_index)]; score = float(scores[local])
        if row["fraud_type"] in fraud_groups: fraud_groups[row["fraud_type"]].append(score)
        if row["hard_negative_type"] != "none": hard_groups.setdefault(row["hard_negative_type"], []).append(score)
    summarize = lambda values: {"n": len(values), "mean_risk_score": float(np.mean(values)),
                                "median_risk_score": float(np.median(values)),
                                "q90_risk_score": float(np.quantile(values, .9))}
    return ({key: summarize(value) for key, value in fraud_groups.items()},
            {key: summarize(value) for key, value in sorted(hard_groups.items())})


def run_training(root: Path = Path(".")) -> dict[str, Any]:
    processed_meta, split_hash = verify_fingerprints(root)
    train = load_split("train", root); validation = load_split("validation", root)
    # TEST se verifica mediante metadata/fingerprint y existencia, pero no se carga.
    assert (root / "data/processed/model_inputs_test.npz").is_file()
    X_train, X_validation = approved_matrix(train), approved_matrix(validation)
    y_train, y_validation = train["y"], validation["y"]
    assert X_train.shape == (len(y_train), len(MODEL_A_FEATURES))
    assert X_validation.shape == (len(y_validation), len(MODEL_A_FEATURES))
    assert np.isfinite(X_train).all() and np.isfinite(X_validation).all()
    assert not any(token in feature.lower() for feature in MODEL_A_FEATURES for token in SUSPICIOUS_TOKENS)
    results, selected = train_candidates(X_train, y_train, X_validation, y_validation)
    model, selected_result = selected["model"], selected["result"]
    train_scores = model.predict_proba(X_train)[:, 1]
    validation_scores = model.predict_proba(X_validation)[:, 1]
    assert np.all((train_scores >= 0) & (train_scores <= 1))
    assert np.all((validation_scores >= 0) & (validation_scores <= 1))
    selected_definition = next(d for d in candidate_definitions(explicit_class_weights(y_train))
                               if d["candidate_id"] == selected_result["candidate_id"])
    replica = build_candidate(selected_definition)
    if selected_definition["algorithm"] == "LogisticRegression":
        replica.fit(X_train, y_train)
    else:
        weights_for_replica = explicit_class_weights(y_train)
        replica.fit(X_train, y_train, sample_weight=np.where(y_train == 1, weights_for_replica[1], weights_for_replica[0]))
    replica_validation_ap = average_precision(y_validation, replica.predict_proba(X_validation)[:, 1])
    output = root / "artefactos/model_a"; experiments = root / "experiments"
    output.mkdir(parents=True, exist_ok=True); experiments.mkdir(parents=True, exist_ok=True)
    model_path = output / "model_a.joblib"; joblib.dump({"estimator": model, "feature_names": MODEL_A_FEATURES}, model_path)
    index_rows = load_analysis_index(root)
    write_scores(output / "train_scores.csv", train["example_index"], index_rows, y_train, train_scores, "train")
    write_scores(output / "validation_scores.csv", validation["example_index"], index_rows, y_validation,
                 validation_scores, "validation")
    with (experiments / "model_a_results.csv").open("w", newline="") as handle:
        fields = ("candidate_id", "algorithm", "parameters", "train_ap", "validation_ap",
                  "train_validation_gap", "fit_time_seconds", "selected")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for result in results:
            writer.writerow({**result, "parameters": json.dumps(result["parameters"], sort_keys=True),
                             "selected": result["candidate_id"] == selected_result["candidate_id"]})
    pr_path = root / "figures/model_a_validation_pr_curve.png"
    save_pr_figure(y_validation, validation_scores, pr_path)
    importance = permutation_importance(model, X_validation, y_validation, scoring="average_precision",
                                        n_repeats=3, random_state=RANDOM_SEED, n_jobs=-1)
    ranked = sorted(zip(MODEL_A_FEATURES, importance.importances_mean, importance.importances_std),
                    key=lambda item: item[1], reverse=True)
    importance_rows = [{"feature": name, "importance_mean": float(mean), "importance_std": float(std)}
                       for name, mean, std in ranked]
    (output / "permutation_importance.json").write_text(json.dumps(importance_rows, indent=2) + "\n")
    top = importance_rows[:15]
    fig, axis = plt.subplots(figsize=(9, 6)); axis.barh([r["feature"] for r in reversed(top)],
        [r["importance_mean"] for r in reversed(top)], xerr=[r["importance_std"] for r in reversed(top)])
    axis.set(title="Importancia por permutación — Modelo A — Validation",
             xlabel="Caída media de Average Precision", ylabel="Feature")
    fig.tight_layout(); importance_path = root / "figures/model_a_permutation_importance.png"
    fig.savefig(importance_path, dpi=150); plt.close(fig)
    fraud_summary, hard_summary = score_group_summary(index_rows, validation["example_index"], validation_scores)
    weights = explicit_class_weights(y_train)
    metadata = {"model_name": "MODEL_A_CANDIDATE", "algorithm": selected_result["algorithm"],
        "candidate_id": selected_result["candidate_id"], "parameters": selected_result["parameters"],
        "random_seed": RANDOM_SEED, "dataset_fingerprint": processed_meta["dataset_source_fingerprint"],
        "processed_dataset_fingerprint": processed_meta["processed_fingerprint"], "split_fingerprint": split_hash,
        "training_split": "TRAIN", "selection_split": "VALIDATION", "test_evaluated": False,
        "n_train": len(y_train), "n_validation": len(y_validation), "n_negative_train": int((y_train == 0).sum()),
        "n_positive_train": int(y_train.sum()), "train_fraud_rate": float(y_train.mean()),
        "validation_fraud_rate": float(y_validation.mean()), "class_weights_from_train": weights,
        "feature_names": list(MODEL_A_FEATURES), "n_features": len(MODEL_A_FEATURES),
        "primary_metric": PRIMARY_METRIC_NAME, "train_ap": float(average_precision(y_train, train_scores)),
        "validation_ap": float(average_precision(y_validation, validation_scores)),
        "train_validation_gap": float(average_precision(y_train, train_scores) - average_precision(y_validation, validation_scores)),
        "reproducibility_validation_ap_run_1": float(average_precision(y_validation, validation_scores)),
        "reproducibility_validation_ap_run_2": replica_validation_ap,
        "reproducibility_absolute_difference": abs(replica_validation_ap - average_precision(y_validation, validation_scores)),
        "selection_rule": "highest validation AP; prefer earlier/simpler candidate within 0.001",
        "validation_fraud_score_summary": fraud_summary, "validation_hard_negative_score_summary": hard_summary,
        "top_permutation_importance": top, "model_fingerprint_sha256": file_sha256(model_path),
        "software_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": "not installed", "scikit-learn": importlib.metadata.version("scikit-learn"),
            "joblib": importlib.metadata.version("joblib"), "platform": platform.platform()},
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (output / "model_a_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"metadata": metadata, "candidate_results": results,
            "artifacts": [str(model_path), str(pr_path), str(importance_path)]}


def main() -> None:
    print(json.dumps(run_training(), indent=2))


if __name__ == "__main__":
    main()
