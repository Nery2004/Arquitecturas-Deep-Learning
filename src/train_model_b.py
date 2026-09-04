"""Entrena Modelo B sobre historia ordenada y operación actual.

No carga agregados ni TEST. Usa TRAIN para fit y VALIDATION para early stopping
y selección. Ejecución: ``.venv/bin/python -m src.train_model_b``.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import RANDOM_SEED
from .evaluation import PRIMARY_METRIC_NAME, average_precision
from .preprocessing import (CURRENT_CATEGORICAL_FEATURES, CURRENT_NUMERIC_FEATURES,
                            MAX_HISTORY, SEQUENCE_CATEGORICAL_FEATURES,
                            SEQUENCE_NUMERIC_FEATURES, file_sha256, processed_fingerprint)
from .sequence_model import SequenceGRU, count_trainable_parameters

SEQUENCE_INPUT_KEYS = ("X_sequence_numeric", "X_sequence_categorical", "sequence_mask",
                       "history_length", "X_current_numeric", "X_current_categorical", "y",
                       "example_index")
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 256
MERCHANT_EMBEDDING_DIM = 6
CHANNEL_EMBEDDING_DIM = 3


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def choose_device() -> torch.device:
    """CPU prioriza reproducibilidad y sigue funcionando en laptops/Colab."""
    return torch.device("cpu")


def load_sequence_split(split: str, root: Path = Path(".")) -> dict[str, np.ndarray]:
    """Carga solo arrays aprobados; X_aggregate ni siquiera se materializa."""
    if split not in {"train", "validation"}:
        raise ValueError("Modelo B solo puede cargar train o validation en esta etapa")
    with np.load(root / f"data/processed/model_inputs_{split}.npz") as archive:
        return {key: archive[key] for key in SEQUENCE_INPUT_KEYS}


class SequenceDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays["y"])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        a = self.arrays
        return (torch.from_numpy(a["X_sequence_numeric"][index]),
                torch.from_numpy(a["X_sequence_categorical"][index].astype(np.int64)),
                torch.tensor(a["history_length"][index], dtype=torch.long),
                torch.from_numpy(a["X_current_numeric"][index]),
                torch.from_numpy(a["X_current_categorical"][index].astype(np.int64)),
                torch.tensor(a["y"][index], dtype=torch.float32))


def verify_fingerprints(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    source = json.loads((root / "data/generated/dataset_metadata.json").read_text())
    processed = json.loads((root / "data/processed/processed_metadata.json").read_text())
    model_a = json.loads((root / "artefactos/model_a/model_a_metadata.json").read_text())
    assert file_sha256(root / "data/generated/transactions.csv") == source["dataset_fingerprint"]
    artifact_dir = root / "artefactos/preprocessing"
    paths = [root / "data/processed/split_config.json", root / "data/processed/example_index.csv",
             root / "data/processed/aggregate_features_raw.csv",
             *(artifact_dir / f"{name}.json" for name in ("aggregate_scaler", "current_scaler",
                                                           "sequence_scaler", "vocabularies")),
             *(root / f"data/processed/model_inputs_{s}.npz" for s in ("train", "validation", "test"))]
    assert processed_fingerprint(paths) == processed["processed_fingerprint"] == model_a["processed_dataset_fingerprint"]
    split_hash = file_sha256(root / "data/processed/split_config.json")
    assert split_hash == model_a["split_fingerprint"]
    return processed, model_a, split_hash


def candidate_definitions() -> list[dict[str, Any]]:
    return [
        {"candidate_id": "b1", "hidden_size": 32, "num_layers": 1, "dense_size": 32,
         "dropout": .2, "learning_rate": 1e-3},
        {"candidate_id": "b2", "hidden_size": 64, "num_layers": 1, "dense_size": 32,
         "dropout": .2, "learning_rate": 1e-3},
        {"candidate_id": "b3", "hidden_size": 64, "num_layers": 1, "dense_size": 64,
         "dropout": .4, "learning_rate": 8e-4},
    ]


def make_loaders(train: dict[str, np.ndarray], validation: dict[str, np.ndarray], seed: int) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    return (DataLoader(SequenceDataset(train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                       generator=generator),
            DataLoader(SequenceDataset(validation), batch_size=BATCH_SIZE, shuffle=False, num_workers=0))


def forward_batch(model: nn.Module, batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_numeric, sequence_categorical, lengths, current_numeric, current_categorical, target = (
        tensor.to(device) for tensor in batch)
    logits = model(sequence_numeric, sequence_categorical, lengths, current_numeric, current_categorical)
    return logits, target


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_function: nn.Module,
             device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval(); losses, scores, targets = [], [], []
    for batch in loader:
        logits, target = forward_batch(model, batch, device)
        losses.append(float(loss_function(logits, target).item()) * len(target))
        scores.append(torch.sigmoid(logits).cpu().numpy()); targets.append(target.cpu().numpy())
    y = np.concatenate(targets); risk = np.concatenate(scores)
    return sum(losses) / len(y), average_precision(y, risk), risk, y


def train_candidate(definition: dict[str, Any], train: dict[str, np.ndarray], validation: dict[str, np.ndarray],
                    vocabularies: dict[str, Any], device: torch.device) -> tuple[nn.Module, list[dict[str, float]], dict[str, Any]]:
    set_reproducible_seed(RANDOM_SEED)
    train_loader, validation_loader = make_loaders(train, validation, RANDOM_SEED)
    model = SequenceGRU(len(vocabularies["merchant_category"]), len(vocabularies["channel"]),
        hidden_size=definition["hidden_size"], num_layers=definition["num_layers"],
        dense_size=definition["dense_size"], dropout=definition["dropout"],
        merchant_embedding_dim=MERCHANT_EMBEDDING_DIM, channel_embedding_dim=CHANNEL_EMBEDDING_DIM).to(device)
    negatives = int((train["y"] == 0).sum()); positives = int(train["y"].sum())
    pos_weight = negatives / positives
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=definition["learning_rate"])
    history, best_state, best_ap, best_epoch, stale = [], None, -1., 0, 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); total_loss = 0.
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, target = forward_batch(model, batch, device)
            loss = loss_function(logits, target); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step(); total_loss += float(loss.item()) * len(target)
        train_loss, train_ap, _, _ = evaluate(model, DataLoader(SequenceDataset(train),
            batch_size=BATCH_SIZE, shuffle=False), loss_function, device)
        validation_loss, validation_ap, _, _ = evaluate(model, validation_loader, loss_function, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                        "train_ap": train_ap, "validation_ap": validation_ap,
                        "learning_rate": definition["learning_rate"]})
        if validation_ap > best_ap + 1e-8:
            best_ap, best_epoch, best_state, stale = validation_ap, epoch, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    train_loss, train_ap, train_scores, _ = evaluate(model, DataLoader(SequenceDataset(train),
        batch_size=BATCH_SIZE, shuffle=False), loss_function, device)
    validation_loss, validation_ap, validation_scores, _ = evaluate(model, validation_loader, loss_function, device)
    result = {**definition, "best_epoch": best_epoch, "epochs_run": len(history),
              "trainable_parameters": count_trainable_parameters(model), "train_loss": train_loss,
              "validation_loss": validation_loss, "train_ap": train_ap, "validation_ap": validation_ap,
              "train_validation_gap": train_ap - validation_ap, "fit_time_seconds": time.perf_counter() - started,
              "pos_weight": pos_weight, "train_scores": train_scores, "validation_scores": validation_scores}
    return model, history, result


def write_scores(path: Path, indices: np.ndarray, index_rows: list[dict[str, str]], y: np.ndarray,
                 scores: np.ndarray, split: str) -> None:
    fields = ("example_id", "transaction_id", "target_timestamp", "y_true", "risk_score", "split",
              "fraud_type", "hard_negative_type")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for local, global_index in enumerate(indices):
            row = index_rows[int(global_index)]
            writer.writerow({"example_id": row["example_id"], "transaction_id": row["transaction_id"],
                "target_timestamp": row["target_timestamp"], "y_true": int(y[local]),
                "risk_score": f"{scores[local]:.17g}", "split": split, "fraud_type": row["fraud_type"],
                "hard_negative_type": row["hard_negative_type"]})


def group_summary(index_rows: list[dict[str, str]], indices: np.ndarray, scores: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    fraud = {name: [] for name in ("testing_cashout", "channel_takeover", "amount_anomaly")}; hard = {}
    for local, global_index in enumerate(indices):
        row = index_rows[int(global_index)]; score = float(scores[local])
        if row["fraud_type"] in fraud: fraud[row["fraud_type"]].append(score)
        if row["hard_negative_type"] != "none": hard.setdefault(row["hard_negative_type"], []).append(score)
    summarize = lambda values: {"n": len(values), "mean_risk_score": float(np.mean(values)),
        "median_risk_score": float(np.median(values)), "q90_risk_score": float(np.quantile(values, .9))}
    return ({k: summarize(v) for k, v in fraud.items()}, {k: summarize(v) for k, v in sorted(hard.items())})


def save_curves(history: list[dict[str, float]], root: Path) -> tuple[Path, Path]:
    epochs = [r["epoch"] for r in history]
    loss_path = root / "figures/model_b_training_loss.png"; ap_path = root / "figures/model_b_training_ap.png"
    fig, axis = plt.subplots(figsize=(8, 5)); axis.plot(epochs, [r["train_loss"] for r in history], label="TRAIN")
    axis.plot(epochs, [r["validation_loss"] for r in history], label="VALIDATION"); axis.set(title="Modelo B — Loss por epoch", xlabel="Epoch", ylabel="BCEWithLogitsLoss"); axis.legend(); fig.tight_layout(); fig.savefig(loss_path,dpi=150);plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5)); axis.plot(epochs, [r["train_ap"] for r in history], label="TRAIN")
    axis.plot(epochs, [r["validation_ap"] for r in history], label="VALIDATION"); axis.set(title="Modelo B — Average Precision por epoch", xlabel="Epoch", ylabel="AP"); axis.legend(); fig.tight_layout(); fig.savefig(ap_path,dpi=150);plt.close(fig)
    return loss_path, ap_path


def run_training(root: Path = Path(".")) -> dict[str, Any]:
    processed, model_a, split_hash = verify_fingerprints(root)
    train, validation = load_sequence_split("train", root), load_sequence_split("validation", root)
    assert len(train["y"]) == model_a["n_train"] and len(validation["y"]) == model_a["n_validation"]
    assert np.isfinite(train["X_sequence_numeric"]).all() and np.isfinite(validation["X_sequence_numeric"]).all()
    assert np.isfinite(train["X_current_numeric"]).all() and np.isfinite(validation["X_current_numeric"]).all()
    assert "X_aggregate" not in train and "X_aggregate" not in validation
    vocabularies = json.loads((root / "artefactos/preprocessing/vocabularies.json").read_text())
    device = choose_device(); trained = []
    for definition in candidate_definitions():
        model, history, result = train_candidate(definition, train, validation, vocabularies, device)
        trained.append({"model": model, "history": history, "result": result})
    best_ap = max(item["result"]["validation_ap"] for item in trained)
    selected = next(item for item in trained if item["result"]["validation_ap"] >= best_ap - .001)
    # Réplica completa del candidato congelado con la misma semilla.
    replica_model, _, replica = train_candidate({k: selected["result"][k] for k in
        ("candidate_id", "hidden_size", "num_layers", "dense_size", "dropout", "learning_rate")},
        train, validation, vocabularies, device)
    del replica_model
    output = root / "artefactos/model_b"; experiments = root / "experiments"
    output.mkdir(parents=True, exist_ok=True); experiments.mkdir(exist_ok=True)
    result = selected["result"]; model = selected["model"]
    checkpoint = {"state_dict": model.state_dict(), "config": {k: result[k] for k in
        ("hidden_size", "num_layers", "dense_size", "dropout")},
        "merchant_vocab_size": len(vocabularies["merchant_category"]),
        "channel_vocab_size": len(vocabularies["channel"]), "merchant_embedding_dim": MERCHANT_EMBEDDING_DIM,
        "channel_embedding_dim": CHANNEL_EMBEDDING_DIM, "feature_names": {
            "sequence_numeric": SEQUENCE_NUMERIC_FEATURES, "sequence_categorical": SEQUENCE_CATEGORICAL_FEATURES,
            "current_numeric": CURRENT_NUMERIC_FEATURES, "current_categorical": CURRENT_CATEGORICAL_FEATURES}}
    model_path = output / "model_b.pt"; torch.save(checkpoint, model_path)
    with (root / "data/processed/example_index.csv").open(newline="") as handle: index_rows = list(csv.DictReader(handle))
    write_scores(output / "train_scores.csv", train["example_index"], index_rows, train["y"], result["train_scores"], "train")
    write_scores(output / "validation_scores.csv", validation["example_index"], index_rows, validation["y"], result["validation_scores"], "validation")
    with (output / "training_history.csv").open("w", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=selected["history"][0].keys(),lineterminator="\n");writer.writeheader();writer.writerows(selected["history"])
    with (experiments / "model_b_results.csv").open("w", newline="") as handle:
        fields=("candidate_id","hidden_size","num_layers","dense_size","dropout","learning_rate","trainable_parameters","best_epoch","epochs_run","train_ap","validation_ap","train_validation_gap","fit_time_seconds","selected")
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for item in trained:
            row={k:v for k,v in item["result"].items() if k in fields};row["selected"]=item is selected;writer.writerow(row)
    loss_path, ap_path = save_curves(selected["history"], root)
    fraud_summary, hard_summary = group_summary(index_rows, validation["example_index"], result["validation_scores"])
    metadata = {"model_name":"MODEL_B_CANDIDATE","architecture":"unidirectional GRU + current branch",
        "candidate_id":result["candidate_id"],"hidden_size":result["hidden_size"],"num_layers":result["num_layers"],
        "dense_size":result["dense_size"],"dropout":result["dropout"],"merchant_embedding_dim":MERCHANT_EMBEDDING_DIM,
        "channel_embedding_dim":CHANNEL_EMBEDDING_DIM,"learning_rate":result["learning_rate"],"batch_size":BATCH_SIZE,
        "optimizer":"AdamW","loss":"BCEWithLogitsLoss","pos_weight":result["pos_weight"],"max_epochs":MAX_EPOCHS,
        "patience":PATIENCE,"best_epoch":result["best_epoch"],"epochs_run":result["epochs_run"],
        "trainable_parameters":result["trainable_parameters"],"random_seed":RANDOM_SEED,"device":str(device),
        "dataset_fingerprint":processed["dataset_source_fingerprint"],"processed_dataset_fingerprint":processed["processed_fingerprint"],
        "split_fingerprint":split_hash,"max_history":MAX_HISTORY,"sequence_numeric_features":list(SEQUENCE_NUMERIC_FEATURES),
        "sequence_categorical_features":list(SEQUENCE_CATEGORICAL_FEATURES),"current_numeric_features":list(CURRENT_NUMERIC_FEATURES),
        "current_categorical_features":list(CURRENT_CATEGORICAL_FEATURES),"aggregate_features_used":False,
        "training_split":"TRAIN","early_stopping_split":"VALIDATION","selection_split":"VALIDATION","test_evaluated":False,
        "n_train":len(train["y"]),"n_validation":len(validation["y"]),"n_positive_train":int(train["y"].sum()),
        "n_negative_train":int((train["y"]==0).sum()),"primary_metric":PRIMARY_METRIC_NAME,"train_ap":result["train_ap"],
        "validation_ap":result["validation_ap"],"train_validation_gap":result["train_validation_gap"],
        "reproducibility_validation_ap_run_1":result["validation_ap"],"reproducibility_validation_ap_run_2":replica["validation_ap"],
        "reproducibility_absolute_difference":abs(result["validation_ap"]-replica["validation_ap"]),
        "model_a_validation_ap_context_only":model_a["validation_ap"],"validation_fraud_score_summary":fraud_summary,
        "validation_hard_negative_score_summary":hard_summary,"model_fingerprint_sha256":file_sha256(model_path),
        "software_versions":{"python":sys.version.split()[0],"numpy":np.__version__,"torch":torch.__version__,
            "scikit-learn":importlib.metadata.version("scikit-learn")},
        "created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (output / "model_b_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    clean_results=[{k:v for k,v in item["result"].items() if k not in ("train_scores","validation_scores")} for item in trained]
    return {"metadata":metadata,"candidate_results":clean_results,"figures":[str(loss_path),str(ap_path)]}


def main() -> None:
    print(json.dumps(run_training(),indent=2))


if __name__ == "__main__": main()
