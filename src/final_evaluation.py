"""Fase 2: única evaluación descriptiva de TEST con decisiones congeladas."""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from .economics import decision_metrics, monthly_normalize
from .evaluation import average_precision, pr_curve
from .hybrid_model import HybridGRU
from .preprocessing import file_sha256
from .sequence_model import SequenceGRU

MODELS = ("A", "B", "C")
BATCH_SIZE = 512
MONTH_DAYS = 30.44


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_test(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / "data/processed/model_inputs_test.npz") as archive:
        return {key: archive[key] for key in archive.files}


def _torch_scores(model: torch.nn.Module, arrays: dict[str, np.ndarray], hybrid: bool) -> np.ndarray:
    model.eval(); chunks = []
    with torch.no_grad():
        for start in range(0, len(arrays["y"]), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(arrays["y"])); sl = slice(start, stop)
            args = [torch.from_numpy(arrays["X_sequence_numeric"][sl]),
                    torch.from_numpy(arrays["X_sequence_categorical"][sl].astype(np.int64)),
                    torch.from_numpy(arrays["history_length"][sl].astype(np.int64)),
                    torch.from_numpy(arrays["X_current_numeric"][sl]),
                    torch.from_numpy(arrays["X_current_categorical"][sl].astype(np.int64))]
            if hybrid:
                args.append(torch.from_numpy(arrays["X_aggregate"][sl]))
            chunks.append(torch.sigmoid(model(*args)).numpy())
    return np.concatenate(chunks)


def score_frozen_models(root: Path, arrays: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, np.ndarray]:
    for model in MODELS:
        ext = "joblib" if model == "A" else "pt"
        path = root / f"artefactos/model_{model.lower()}/model_{model.lower()}.{ext}"
        assert file_sha256(path) == config["model_checkpoints_sha256"][model]
    # A fue congelado con sklearn 1.9/NumPy 2.4; se puntúa en ese mismo entorno.
    with tempfile.TemporaryDirectory(prefix="model_a_test_scores_") as temporary:
        output = Path(temporary) / "scores.npy"
        subprocess.run(["python3", "-m", "src.score_model_a_test", str(root.resolve()), str(output)],
                       cwd=root, check=True)
        result = {"A": np.load(output, allow_pickle=False)}
    checkpoint_b = torch.load(root / "artefactos/model_b/model_b.pt", map_location="cpu")
    model_b = SequenceGRU(checkpoint_b["merchant_vocab_size"], checkpoint_b["channel_vocab_size"],
                          merchant_embedding_dim=checkpoint_b["merchant_embedding_dim"],
                          channel_embedding_dim=checkpoint_b["channel_embedding_dim"], **checkpoint_b["config"])
    model_b.load_state_dict(checkpoint_b["state_dict"])
    result["B"] = _torch_scores(model_b, arrays, False)
    checkpoint_c = torch.load(root / "artefactos/model_c/model_c.pt", map_location="cpu")
    cc = checkpoint_c["config"]
    model_c = HybridGRU(checkpoint_c["merchant_vocab_size"], checkpoint_c["channel_vocab_size"],
                        aggregate_size=checkpoint_c["aggregate_size"], aggregate_hidden=cc["aggregate_hidden"],
                        fusion_hidden=cc["fusion_hidden"], dropout=cc["dropout"])
    model_c.load_state_dict(checkpoint_c["state_dict"])
    result["C"] = _torch_scores(model_c, arrays, True)
    return result


def _index_rows(root: Path) -> list[dict[str, str]]:
    with (root / "data/processed/example_index.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _transactions(root: Path) -> dict[str, dict[str, str]]:
    with (root / "data/generated/transactions.csv").open(newline="") as handle:
        return {row["transaction_id"]: row for row in csv.DictReader(handle)}


def _write_scores(root: Path, model: str, arrays: dict[str, np.ndarray], index: list[dict[str, str]],
                  scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    fields = ("example_id", "transaction_id", "target_timestamp", "y_true", "risk_score", "threshold",
              "predicted_label", "fraud_type", "hard_negative_type", "history_length")
    rows = []
    for local, global_i in enumerate(arrays["example_index"]):
        source = index[int(global_i)]
        rows.append({"example_id": source["example_id"], "transaction_id": source["transaction_id"],
                     "target_timestamp": source["target_timestamp"], "y_true": int(arrays["y"][local]),
                     "risk_score": float(scores[local]), "threshold": threshold,
                     "predicted_label": int(scores[local] >= threshold), "fraud_type": source["fraud_type"],
                     "hard_negative_type": source["hard_negative_type"],
                     "history_length": int(arrays["history_length"][local])})
    path = root / f"artefactos/model_{model.lower()}/test_scores.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    return rows


def _mechanisms(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in ("testing_cashout", "channel_takeover", "amount_anomaly"):
        group = [r for r in rows if r["fraud_type"] == name]
        detected = sum(r["predicted_label"] for r in group)
        result[name] = {"n": len(group), "detected": detected, "missed": len(group) - detected,
                        "recall": detected / len(group) if group else 0.0,
                        "mean_risk_score": float(np.mean([r["risk_score"] for r in group])) if group else None,
                        "median_risk_score": float(np.median([r["risk_score"] for r in group])) if group else None}
    return result


def _representative(rows: list[dict[str, Any]], transactions: dict[str, dict[str, str]], false_negative: bool) -> list[dict[str, Any]]:
    predicate = ((lambda r: r["y_true"] == 1 and r["predicted_label"] == 0) if false_negative else
                 (lambda r: r["y_true"] == 0 and r["predicted_label"] == 1))
    subset = [r for r in rows if predicate(r)]
    chosen = []
    group_field = "fraud_type" if false_negative else "hard_negative_type"
    names = sorted({r[group_field] for r in subset})
    for name in names:
        group = sorted((r for r in subset if r[group_field] == name), key=lambda r: r["risk_score"])
        for position in sorted({0, len(group) // 2, len(group) - 1}):
            if len(chosen) >= 10: break
            row = group[position]
            if row not in chosen: chosen.append(row)
    if len(chosen) < min(8, len(subset)):
        ordered = sorted(subset, key=lambda r: r["risk_score"])
        for position in np.linspace(0, len(ordered) - 1, min(10, len(ordered)), dtype=int):
            if ordered[position] not in chosen: chosen.append(ordered[position])
            if len(chosen) >= 10: break
    output = []
    for row in chosen[:10]:
        tx = transactions[row["transaction_id"]]
        output.append({key: row[key] for key in ("transaction_id", "target_timestamp", "fraud_type", "hard_negative_type", "history_length", "risk_score", "threshold")} |
                      {"amount": float(tx["amount"]), "channel": tx["channel"], "merchant": tx["merchant_category"]})
    return output


def _write_error_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def _figures(root: Path, y: np.ndarray, scores: dict[str, np.ndarray], metrics: dict[str, Any],
             candidate: str, mechanisms: dict[str, Any], monthly: dict[str, float]) -> list[str]:
    paths = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODELS:
        precision, recall, _ = pr_curve(y, scores[model]); ax.plot(recall, precision, label=f"{model} (AP={metrics[model]['test_ap']:.3f})")
    ax.set(title="Precision–Recall — TEST (descriptiva)", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1)); ax.legend(); fig.tight_layout()
    p = root / "figures/test_pr_curves_abc.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))
    fig, ax = plt.subplots(figsize=(7, 5)); ax.bar(MODELS, [metrics[m]["economic_cost"] for m in MODELS], color=("#5b8c5a", "#3b82a0", "#7251a3")); ax.set(title="Costo económico — TEST", xlabel="Modelo", ylabel="Costo (Q)"); fig.tight_layout()
    p = root / "figures/test_economic_cost_abc.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))
    cm = np.array([[metrics[candidate]["TN"], metrics[candidate]["FP"]], [metrics[candidate]["FN"], metrics[candidate]["TP"]]])
    fig, ax = plt.subplots(figsize=(6, 5)); ConfusionMatrixDisplay(cm, display_labels=("Legítima", "Fraude")).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False); ax.set_title(f"Matriz de confusión — TEST — Modelo {candidate}"); fig.tight_layout()
    p = root / "figures/test_confusion_matrix_final_candidate.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))
    fig, ax = plt.subplots(figsize=(8, 5)); names=list(mechanisms); ax.bar(names, [mechanisms[n]["recall"] for n in names], color="#5b8c5a"); ax.set(title=f"Recall por mecanismo — TEST — Modelo {candidate}", ylabel="Recall", ylim=(0, 1)); ax.tick_params(axis="x", rotation=15); fig.tight_layout()
    p = root / "figures/test_recall_by_mechanism_candidate.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))
    fig, ax = plt.subplots(figsize=(7, 5)); ax.bar(("A", f"Candidato {candidate}"), (monthly["A"], monthly["candidate"]), color=("#5b8c5a", "#7251a3")); ax.set(title="Costo mensual equivalente — escala simulada", ylabel="Costo (Q/mes)"); fig.tight_layout()
    p = root / "figures/monthly_economic_comparison.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))
    return paths


def run_final_evaluation(root: Path = Path(".")) -> dict[str, Any]:
    result_path = root / "artefactos/final_evaluation.json"
    if result_path.exists(): raise RuntimeError("TEST ya fue evaluado; el experimento está cerrado")
    config_path = root / "artefactos/final_decision_config.json"; frozen_hash = file_sha256(config_path)
    config = _json(config_path)
    assert config["test_opened"] is False and config["selection_data"] == "VALIDATION only"
    frozen_fields = {k: config[k] for k in ("candidate_model", "final_threshold", "candidate_checkpoint_sha256",
                                             "validation_thresholds", "model_checkpoints_sha256")}
    arrays = _load_test(root)
    scores = score_frozen_models(root, arrays, config); y = arrays["y"]
    assert all(len(s) == len(y) and np.isfinite(s).all() and np.logical_and(s >= 0, s <= 1).all() for s in scores.values())
    index = _index_rows(root); rows = {}; metrics = {}
    for model in MODELS:
        threshold = config["validation_thresholds"][model]
        dm = decision_metrics(y, scores[model], threshold)
        metrics[model] = {"test_ap": average_precision(y, scores[model]), **dm.as_dict()}
        rows[model] = _write_scores(root, model, arrays, index, scores[model], threshold)
        assert dm.TP + dm.FP + dm.TN + dm.FN == len(y)
    timestamps = np.array([np.datetime64(index[int(i)]["target_timestamp"]) for i in arrays["example_index"]])
    test_days = float((timestamps.max() - timestamps.min()) / np.timedelta64(1, "D"))
    candidate = config["candidate_model"]
    monthly = {"A": monthly_normalize(metrics["A"]["economic_cost"], test_days),
               "candidate": monthly_normalize(metrics[candidate]["economic_cost"], test_days)}
    mechanisms = _mechanisms(rows[candidate]); transactions = _transactions(root)
    false_negatives = _representative(rows[candidate], transactions, True)
    false_positives = _representative(rows[candidate], transactions, False)
    _write_error_csv(root / "experiments/final_candidate_false_negatives.csv", false_negatives)
    _write_error_csv(root / "experiments/final_candidate_false_positives.csv", false_positives)
    never_cost = int(y.sum()) * 4200; everything_cost = int((y == 0).sum()) * 180
    figures = _figures(root, y, scores, metrics, candidate, mechanisms, monthly)
    evaluated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    final = {"schema_version": "1.0", "experimental_status": "CLOSED", "test_opened_once": True,
             "test_evaluation_timestamp": evaluated_at, "frozen_config_sha256_before_test": frozen_hash,
             "dataset_fingerprint": config["dataset_fingerprint"], "processed_fingerprint": config["processed_fingerprint"],
             "split_fingerprint": config["split_fingerprint"], "test_start": str(timestamps.min()), "test_end": str(timestamps.max()),
             "test_days": test_days, "monthly_normalization_days": MONTH_DAYS, "models": metrics,
             "validation_thresholds": config["validation_thresholds"], "candidate_model": candidate,
             "candidate_threshold": config["final_threshold"], "test_baselines": {"never_block_cost": never_cost, "block_everything_cost": everything_cost},
             "savings_vs_A": metrics["A"]["economic_cost"] - metrics[candidate]["economic_cost"],
             "monthly_cost_A": monthly["A"], "monthly_cost_candidate": monthly["candidate"],
             "monthly_savings_vs_A": monthly["A"] - monthly["candidate"], "scale_note": "Equivalente mensual dentro de la escala simulada; no extrapolar directamente a 1.4 millones de tarjetas.",
             "candidate_mechanisms": mechanisms, "representative_false_negatives": false_negatives,
             "representative_false_positives": false_positives, "figures": figures,
             "execution_incidents": [{"stage": "initial TEST scoring attempt", "impact": "No metrics or model decisions were produced or changed.",
                 "detail": "The PyTorch environment could not deserialize Model A because it uses older NumPy/scikit-learn versions.",
                 "resolution": "Model A inference ran under its frozen sklearn 1.9/NumPy 2.4 environment; B/C ran under their frozen PyTorch environment."}],
             "recommendation": "CONSERVAR", "recommendation_reason": "A fue el candidato prefijado y las alternativas secuenciales no redujeron el costo frente a A.",
             "integrity_statement": "El conjunto TEST fue utilizado únicamente después de congelar modelos, selección del candidato y thresholds. Ninguna decisión de modelado o threshold fue modificada a partir de sus resultados."}
    result_path.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
    config["test_opened"] = True; config["test_evaluation_timestamp"] = evaluated_at
    config["pre_test_config_sha256"] = frozen_hash; config["experimental_status"] = "CLOSED"
    assert all(config[k] == value for k, value in frozen_fields.items())
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return final


def main() -> None:
    print(json.dumps(run_final_evaluation(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
