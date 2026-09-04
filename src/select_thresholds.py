"""Fase 1: audita y congela modelo/thresholds usando solo VALIDATION."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .economics import (FALSE_NEGATIVE_COST_GTQ, FALSE_POSITIVE_COST_GTQ,
                        decision_metrics, select_economic_threshold)
from .preprocessing import file_sha256, processed_fingerprint

MODELS = ("A", "B", "C")
FINAL_CANDIDATE_MODEL = "A"
SELECTION_REASON = ("A obtuvo el mayor AP congelado en VALIDATION (0.9100 frente a 0.8152 de C y "
                    "0.7207 de B) y también el menor costo económico de VALIDATION. B sí mostró "
                    "valor del orden bajo permutación, y C mejoró B, pero ninguno superó al sistema "
                    "agregado A en la evidencia usada antes de TEST.")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _processed_paths(root: Path) -> list[Path]:
    prep = root / "artefactos/preprocessing"
    return [root / "data/processed/split_config.json", root / "data/processed/example_index.csv",
            root / "data/processed/aggregate_features_raw.csv",
            *(prep / f"{name}.json" for name in ("aggregate_scaler", "current_scaler", "sequence_scaler", "vocabularies")),
            *(root / f"data/processed/model_inputs_{split}.npz" for split in ("train", "validation", "test"))]


def audit_frozen_state(root: Path) -> dict[str, Any]:
    dataset = _json(root / "data/generated/dataset_metadata.json")
    processed = _json(root / "data/processed/processed_metadata.json")
    metas = {m: _json(root / f"artefactos/model_{m.lower()}/model_{m.lower()}_metadata.json") for m in MODELS}
    falsification = _json(root / "artefactos/model_b/falsification_metadata.json")
    assert file_sha256(root / "data/generated/transactions.csv") == dataset["dataset_fingerprint"]
    assert processed_fingerprint(_processed_paths(root)) == processed["processed_fingerprint"]
    split_hash = file_sha256(root / "data/processed/split_config.json")
    for model in MODELS:
        ext = "joblib" if model == "A" else "pt"
        checkpoint = root / f"artefactos/model_{model.lower()}/model_{model.lower()}.{ext}"
        assert file_sha256(checkpoint) == metas[model]["model_fingerprint_sha256"]
        assert metas[model]["test_evaluated"] is False
        assert metas[model]["split_fingerprint"] == split_hash
    assert falsification["evaluation_split"] == "VALIDATION" and falsification["test_evaluated"] is False
    return {"dataset_fingerprint": dataset["dataset_fingerprint"],
            "processed_fingerprint": processed["processed_fingerprint"], "split_fingerprint": split_hash,
            "metadata": metas, "falsification": falsification}


def load_validation_scores(root: Path, model: str) -> dict[str, np.ndarray]:
    path = root / f"artefactos/model_{model.lower()}/validation_scores.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["split"].lower() == "validation" for row in rows)
    return {"example_id": np.array([r["example_id"] for r in rows]),
            "y": np.array([int(r["y_true"]) for r in rows], dtype=np.int8),
            "score": np.array([float(r["risk_score"]) for r in rows], dtype=float)}


def _write_threshold_table(path: Path, rows: list[Any]) -> None:
    fields = ("model", "threshold", "TP", "FP", "TN", "FN", "precision", "recall", "F1", "economic_cost")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for model, model_rows in rows:
            for metrics in model_rows:
                writer.writerow({"model": model, **metrics.as_dict()})


def _figures(root: Path, analyses: dict[str, list[Any]], selected: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for model in MODELS:
        rows = analyses[model]
        ax.plot([r.threshold for r in rows], [r.economic_cost for r in rows], label=model, alpha=.8)
        best = selected[model]; ax.scatter([best.threshold], [best.economic_cost], s=55)
    final = selected[FINAL_CANDIDATE_MODEL]
    ax.axvline(final.threshold, color="black", linestyle="--", alpha=.6,
               label=f"Final A: {final.threshold:.6g}, Q{final.economic_cost:,}")
    ax.set(title="Economic Cost vs Threshold — Validation", xlabel="Threshold", ylabel="Costo (Q)", xlim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(root / "figures/economic_cost_vs_threshold_validation.png", dpi=150); plt.close(fig)
    rows = analyses[FINAL_CANDIDATE_MODEL]
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, label in (("precision", "Precision"), ("recall", "Recall"), ("F1", "F1")):
        ax.plot([r.threshold for r in rows], [getattr(r, key) for r in rows], label=label)
    ax.axvline(final.threshold, color="black", linestyle="--", label=f"Threshold económico {final.threshold:.6g}")
    ax.set(title="Precision, Recall y F1 vs Threshold — Validation — Modelo A", xlabel="Threshold", ylabel="Métrica", xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(root / "figures/decision_metrics_vs_threshold_validation.png", dpi=150); plt.close(fig)


def freeze_validation_decisions(root: Path = Path(".")) -> dict[str, Any]:
    if (root / "artefactos/final_evaluation.json").exists():
        raise RuntimeError("La evaluación final ya existe; el experimento está cerrado")
    audit = audit_frozen_state(root)
    scores = {model: load_validation_scores(root, model) for model in MODELS}
    for model in MODELS[1:]:
        assert np.array_equal(scores["A"]["example_id"], scores[model]["example_id"])
        assert np.array_equal(scores["A"]["y"], scores[model]["y"])
    assert all(np.isfinite(v["score"]).all() and np.logical_and(v["score"] >= 0, v["score"] <= 1).all() for v in scores.values())
    selected, analyses = {}, {}
    for model in MODELS:
        selected[model], analyses[model] = select_economic_threshold(scores[model]["y"], scores[model]["score"])
    assert selected["A"].economic_cost <= selected["B"].economic_cost
    assert selected["A"].economic_cost <= selected["C"].economic_cost
    _write_threshold_table(root / "experiments/threshold_analysis.csv", list(analyses.items()))
    _figures(root, analyses, selected)
    y = scores["A"]["y"]
    never = decision_metrics(y, np.zeros(len(y)), np.nextafter(0.0, np.inf))
    everything = decision_metrics(y, np.zeros(len(y)), 0.0)
    checkpoint_rel = "artefactos/model_a/model_a.joblib"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config = {"schema_version": "1.0", "frozen_before_test_at_utc": now, "test_opened": False,
              "test_evaluation_timestamp": None, "candidate_model": FINAL_CANDIDATE_MODEL,
              "final_threshold": selected[FINAL_CANDIDATE_MODEL].threshold,
              "selection_reason": SELECTION_REASON, "selection_data": "VALIDATION only",
              "tie_break_rule": "highest threshold among exact minimum-cost ties",
              "primary_metric": "AUC-PR / Average Precision (AP)",
              "economic_costs_gtq": {"false_negative": FALSE_NEGATIVE_COST_GTQ, "false_positive": FALSE_POSITIVE_COST_GTQ},
              "economic_interpretation": "Academic expected-damage approximation; a TP is not guaranteed to save exactly Q4,200.",
              "dataset_fingerprint": audit["dataset_fingerprint"], "processed_fingerprint": audit["processed_fingerprint"],
              "split_fingerprint": audit["split_fingerprint"], "candidate_checkpoint": checkpoint_rel,
              "candidate_checkpoint_sha256": audit["metadata"]["A"]["model_fingerprint_sha256"],
              "model_checkpoints_sha256": {m: audit["metadata"][m]["model_fingerprint_sha256"] for m in MODELS},
              "validation_ap": {m: audit["metadata"][m]["validation_ap"] for m in MODELS},
              "validation_thresholds": {m: selected[m].threshold for m in MODELS},
              "validation_metrics_at_threshold": {m: selected[m].as_dict() for m in MODELS},
              "validation_baselines": {"never_block": never.as_dict(), "block_everything": everything.as_dict()},
              "falsification_summary": {"permuted_mean_ap": audit["falsification"]["permuted_mean_ap"],
                  "permutation_relative_drop": audit["falsification"]["permutation_relative_drop"],
                  "short_history_ap": audit["falsification"]["short_history_ap"]},
              "immutability_statement": "No se realizarán cambios posteriores basados en TEST."}
    path = root / "artefactos/final_decision_config.json"
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    config["frozen_config_sha256"] = file_sha256(path)
    print("=" * 40 + "\nFINAL DECISIONS FROZEN BEFORE TEST\n" + "=" * 40)
    print(f"Candidate model: {config['candidate_model']}\nThreshold: {config['final_threshold']}\nDataset fingerprint: {config['dataset_fingerprint']}\nProcessed fingerprint: {config['processed_fingerprint']}\nSplit fingerprint: {config['split_fingerprint']}\nCheckpoint: {checkpoint_rel}\nPrimary metric: {config['primary_metric']}\nEconomic costs: FN Q4,200 / FP Q180\nValidation AP: {config['validation_ap']['A']}\nValidation economic cost: Q{selected['A'].economic_cost:,}\n\nNo se realizarán cambios posteriores basados en TEST.")
    return config


def main() -> None:
    print(json.dumps(freeze_validation_decisions(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
