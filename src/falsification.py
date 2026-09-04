"""Falsificaciones de Modelo B sobre VALIDATION, sin reentrenar ni abrir TEST."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluation import average_precision
from .preprocessing import file_sha256
from .sequence_model import SequenceGRU
from .train_model_b import BATCH_SIZE, SequenceDataset, evaluate, load_sequence_split

PERMUTATION_SEEDS = tuple(range(100, 110))
SHORT_HISTORY = 3


def permute_valid_history(arrays: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    """Permuta eventos completos válidos; mantiene PAD y cada variable pegada al evento."""
    result = {key: value.copy() for key, value in arrays.items()}
    numeric = result["X_sequence_numeric"]
    categorical = result["X_sequence_categorical"]
    width = numeric.shape[1]
    for index, length_value in enumerate(result["history_length"]):
        length = int(length_value)
        if length <= 1:
            continue
        start = width - length
        order = np.random.default_rng(np.random.SeedSequence([seed, index])).permutation(length)
        numeric[index, start:] = numeric[index, start:][order]
        categorical[index, start:] = categorical[index, start:][order]
    return result


def truncate_history(arrays: dict[str, np.ndarray], max_events: int = SHORT_HISTORY) -> dict[str, np.ndarray]:
    """Conserva los eventos reales más recientes y mantiene shape mediante left padding."""
    result = {key: value.copy() for key, value in arrays.items()}
    width = result["X_sequence_numeric"].shape[1]
    new_lengths = np.minimum(result["history_length"], max_events).astype(result["history_length"].dtype)
    for index, new_length_value in enumerate(new_lengths):
        new_length = int(new_length_value)
        cutoff = width - new_length
        result["X_sequence_numeric"][index, :cutoff] = 0
        result["X_sequence_categorical"][index, :cutoff] = 0
        result["sequence_mask"][index, :cutoff] = False
        result["sequence_mask"][index, cutoff:] = True
    result["history_length"] = new_lengths
    return result


def load_frozen_model(root: Path, metadata: dict[str, Any]) -> SequenceGRU:
    checkpoint = torch.load(root / "artefactos/model_b/model_b.pt", map_location="cpu")
    model = SequenceGRU(checkpoint["merchant_vocab_size"], checkpoint["channel_vocab_size"],
        **checkpoint["config"], merchant_embedding_dim=checkpoint["merchant_embedding_dim"],
        channel_embedding_dim=checkpoint["channel_embedding_dim"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    assert file_sha256(root / "artefactos/model_b/model_b.pt") == metadata["model_fingerprint_sha256"]
    return model


def predict(model: SequenceGRU, arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Obtiene scores continuos; la loss dummy no interviene en AP."""
    loss = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(SequenceDataset(arrays), batch_size=BATCH_SIZE, shuffle=False)
    _, _, scores, targets = evaluate(model, loader, loss, torch.device("cpu"))
    assert np.array_equal(targets.astype(np.int8), arrays["y"])
    return scores


def load_index(root: Path) -> list[dict[str, str]]:
    with (root / "data/processed/example_index.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def mechanism_ap(index_rows: list[dict[str, str]], global_indices: np.ndarray,
                 scores: np.ndarray) -> dict[str, float]:
    """AP one-vs-legitimate: excluye otros mecanismos fraudulentos."""
    output = {}
    for mechanism in ("testing_cashout", "channel_takeover", "amount_anomaly"):
        keep, labels = [], []
        for local, global_index in enumerate(global_indices):
            kind = index_rows[int(global_index)]["fraud_type"]
            if kind == "legitimate" or kind == mechanism:
                keep.append(local); labels.append(int(kind == mechanism))
        output[mechanism] = average_precision(np.asarray(labels), scores[np.asarray(keep)])
    return output


def grouped_scores(index_rows: list[dict[str, str]], global_indices: np.ndarray,
                   scores: np.ndarray, field: str, allowed: tuple[str, ...]) -> dict[str, dict[str, float]]:
    groups = {name: [] for name in allowed}
    for local, global_index in enumerate(global_indices):
        name = index_rows[int(global_index)][field]
        if name in groups: groups[name].append(float(scores[local]))
    return {name: {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
                   "q90": float(np.quantile(values, .9))} for name, values in groups.items()}


def permutation_order(length: int, seed: int, local_index: int) -> list[int]:
    return np.random.default_rng(np.random.SeedSequence([seed, local_index])).permutation(length).tolist()


def real_example(root: Path, validation: dict[str, np.ndarray], index_rows: list[dict[str, str]],
                 seed: int = PERMUTATION_SEEDS[0]) -> dict[str, Any]:
    with (root / "data/generated/transactions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_card: dict[str, list[dict[str, str]]] = {}
    for row in rows: by_card.setdefault(row["card_id"], []).append(row)
    local = next(i for i, global_index in enumerate(validation["example_index"])
                 if index_rows[int(global_index)]["fraud_type"] == "testing_cashout" and
                 int(validation["history_length"][i]) >= 6)
    example = index_rows[int(validation["example_index"][local])]
    target_key = (example["target_timestamp"], example["transaction_id"])
    history = [r for r in by_card[example["card_id"]]
               if (r["timestamp"], r["transaction_id"]) < target_key][-int(validation["history_length"][local]):]
    fields = ("timestamp", "amount", "channel", "merchant_category")
    readable = [{field: row[field] for field in fields} for row in history]
    order = permutation_order(len(history), seed, local)
    target = next(row for row in by_card[example["card_id"]] if row["transaction_id"] == example["transaction_id"])
    return {"example_id": example["example_id"], "permutation_seed": seed, "original": readable,
            "permuted": [readable[i] for i in order], "short_history": readable[-SHORT_HISTORY:],
            "target": {field: target[field] for field in ("transaction_id", "timestamp", "amount",
                                                           "channel", "merchant_category", "is_fraud")}}


def save_figures(original_ap: float, permutation_aps: list[float], short_ap: float,
                 model_a_ap: float, root: Path) -> list[str]:
    paths = []
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter([0] * len(permutation_aps), permutation_aps, alpha=.75, label="10 permutaciones")
    axis.scatter([0], [original_ap], marker="D", s=90, label="B original")
    axis.set(xticks=[0], xticklabels=["VALIDATION"], ylabel="Average Precision",
             title="Orden original vs. historias permutadas"); axis.legend(); fig.tight_layout()
    path=root/"figures/order_permutation_validation.png";fig.savefig(path,dpi=150);plt.close(fig);paths.append(str(path))
    fig, axis = plt.subplots(figsize=(8, 5));axis.bar(("B original","B historia ≤3"),(original_ap,short_ap),color=("#3b82a0","#d39c43"));axis.set(ylabel="Average Precision",title="Efecto de recortar el historial en VALIDATION",ylim=(0,1));fig.tight_layout();path=root/"figures/history_truncation_validation.png";fig.savefig(path,dpi=150);plt.close(fig);paths.append(str(path))
    fig, axis = plt.subplots(figsize=(9, 5));axis.bar(("A agregados","B original","B permutado","B historia ≤3"),(model_a_ap,original_ap,float(np.mean(permutation_aps)),short_ap),color=("#5b8c5a","#3b82a0","#c75c5c","#d39c43"));axis.set(ylabel="Average Precision",title="Resumen de evidencia en VALIDATION",ylim=(0,1));fig.tight_layout();path=root/"figures/order_evidence_summary.png";fig.savefig(path,dpi=150);plt.close(fig);paths.append(str(path))
    return paths


def run_falsifications(root: Path = Path(".")) -> dict[str, Any]:
    model_b_meta = json.loads((root/"artefactos/model_b/model_b_metadata.json").read_text())
    model_a_meta = json.loads((root/"artefactos/model_a/model_a_metadata.json").read_text())
    processed_meta = json.loads((root/"data/processed/processed_metadata.json").read_text())
    checkpoint_hash_before = file_sha256(root/"artefactos/model_b/model_b.pt")
    assert checkpoint_hash_before == model_b_meta["model_fingerprint_sha256"]
    assert model_b_meta["dataset_fingerprint"] == processed_meta["dataset_source_fingerprint"]
    assert model_b_meta["processed_dataset_fingerprint"] == processed_meta["processed_fingerprint"]
    validation = load_sequence_split("validation", root)
    model = load_frozen_model(root, model_b_meta); index_rows = load_index(root)
    original_scores = predict(model, validation); original_ap = average_precision(validation["y"], original_scores)
    assert abs(original_ap - model_b_meta["validation_ap"]) < 1e-12
    original_mechanism = mechanism_ap(index_rows, validation["example_index"], original_scores)
    permutation_runs = []
    for seed in PERMUTATION_SEEDS:
        changed = permute_valid_history(validation, seed)
        scores = predict(model, changed)
        permutation_runs.append({"seed": seed, "validation_ap": average_precision(validation["y"], scores),
                                 "mechanism_ap": mechanism_ap(index_rows, validation["example_index"], scores)})
    permutation_aps = [run["validation_ap"] for run in permutation_runs]
    short = truncate_history(validation); short_scores = predict(model, short)
    short_ap = average_precision(validation["y"], short_scores)
    short_mechanism = mechanism_ap(index_rows, validation["example_index"], short_scores)
    hard_types = ("legitimate_trip","legitimate_large_purchase","legitimate_shopping_spree","legitimate_micro_purchases")
    hard_summary = {"original": grouped_scores(index_rows,validation["example_index"],original_scores,"hard_negative_type",hard_types),
                    "permuted_seed_100": grouped_scores(index_rows,validation["example_index"],predict(model,permute_valid_history(validation,100)),"hard_negative_type",hard_types),
                    "short": grouped_scores(index_rows,validation["example_index"],short_scores,"hard_negative_type",hard_types)}
    figures = save_figures(original_ap, permutation_aps, short_ap, model_a_meta["validation_ap"], root)
    metadata = {"model_b_checkpoint_hash":checkpoint_hash_before,"dataset_fingerprint":model_b_meta["dataset_fingerprint"],
        "processed_fingerprint":model_b_meta["processed_dataset_fingerprint"],"split_fingerprint":model_b_meta["split_fingerprint"],
        "evaluation_split":"VALIDATION","test_evaluated":False,"original_validation_ap":original_ap,
        "permutation_seeds":list(PERMUTATION_SEEDS),"permutation_runs":permutation_runs,
        "permuted_mean_ap":float(np.mean(permutation_aps)),"permuted_std_ap":float(np.std(permutation_aps)),
        "permuted_min_ap":float(np.min(permutation_aps)),"permuted_max_ap":float(np.max(permutation_aps)),
        "permutation_absolute_drop":original_ap-float(np.mean(permutation_aps)),
        "permutation_relative_drop":(original_ap-float(np.mean(permutation_aps)))/original_ap,
        "original_mechanism_ap_one_vs_legitimate":original_mechanism,"short_history_max_events":SHORT_HISTORY,
        "short_history_ap":short_ap,"short_history_absolute_drop":original_ap-short_ap,
        "short_history_relative_drop":(original_ap-short_ap)/original_ap,
        "short_mechanism_ap_one_vs_legitimate":short_mechanism,
        "history_length_groups":{"unchanged_le_3":int((validation["history_length"]<=3).sum()),
                                 "truncated_gt_3":int((validation["history_length"]>3).sum())},
        "hard_negative_score_summary":hard_summary,"real_example":real_example(root,validation,index_rows),
        "figures":figures,"checkpoint_hash_after":file_sha256(root/"artefactos/model_b/model_b.pt"),
        "created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds")}
    assert metadata["checkpoint_hash_after"] == checkpoint_hash_before
    output=root/"artefactos/model_b/falsification_metadata.json";output.write_text(json.dumps(metadata,indent=2)+"\n")
    with (root/"experiments/falsification_results.csv").open("w",newline="") as handle:
        fields=("condition","seed","validation_ap","difference_vs_original")
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n");writer.writeheader()
        writer.writerow({"condition":"B original","seed":"","validation_ap":original_ap,"difference_vs_original":0})
        for run in permutation_runs:writer.writerow({"condition":"B permuted","seed":run["seed"],"validation_ap":run["validation_ap"],"difference_vs_original":run["validation_ap"]-original_ap})
        writer.writerow({"condition":"B short history","seed":"","validation_ap":short_ap,"difference_vs_original":short_ap-original_ap})
    return metadata


def main() -> None:
    print(json.dumps(run_falsifications(),indent=2))


if __name__ == "__main__": main()
