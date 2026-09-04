"""Partición temporal y preprocessing causal del Dataset Version 1.

Se ejecuta con ``python3 -m src.preprocessing``. No entrena modelos: crea el
índice maestro, agregados, secuencias, transformaciones ajustadas con TRAIN y
artefactos reproducibles para A, B y C.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PREPROCESSING_VERSION = "1.0.0"
MIN_HISTORY = 1
MAX_HISTORY = 12
PAD_INDEX = 0
UNK_INDEX = 1
FORBIDDEN_FEATURE_NAMES = {
    "transaction_id", "card_id", "is_fraud", "fraud_type", "fraud_stage",
    "hard_negative_type", "customer_profile",
}
CURRENT_NUMERIC_FEATURES = (
    "current_amount", "current_distance_from_home_km", "current_is_international",
    "current_hour_sin", "current_hour_cos", "current_day_sin", "current_day_cos",
    "time_since_previous_hours",
)
CURRENT_CATEGORICAL_FEATURES = ("current_merchant_category", "current_channel")
SEQUENCE_NUMERIC_FEATURES = (
    "amount", "distance_from_home_km", "is_international", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "time_since_previous_hours",
)
SEQUENCE_CATEGORICAL_FEATURES = ("merchant_category", "channel")
AGGREGATE_FEATURES = (
    "historical_mean_amount", "historical_std_amount", "historical_max_amount",
    "historical_transaction_count", "current_vs_mean_amount_ratio", "amount_zscore_personal",
    "historical_mean_distance", "distance_ratio_to_typical", "current_channel_frequency_history",
    "current_channel_is_unusual", "current_merchant_frequency_history", "current_merchant_is_unusual",
    "tx_count_1h", "tx_count_6h", "tx_count_24h", "tx_count_7d", "amount_mean_24h",
    "amount_max_24h", "amount_sum_24h", "merchant_diversity_24h", "channel_diversity_24h",
    "international_count_24h", "has_short_history",
)


@dataclass
class Standardizer:
    """Imputación por mediana y estandarización aprendidas solo con TRAIN."""

    feature_names: tuple[str, ...]
    medians: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    fitted_split: str | None = None

    def fit(self, values: np.ndarray, split: str = "train") -> "Standardizer":
        assert split == "train", "Los preprocessors solo pueden ajustarse con TRAIN"
        self.medians = np.nanmedian(values, axis=0)
        clean = np.where(np.isnan(values), self.medians, values)
        self.means = clean.mean(axis=0)
        scales = clean.std(axis=0)
        self.scales = np.where(scales < 1e-8, 1.0, scales)
        self.fitted_split = split
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        assert self.fitted_split == "train"
        clean = np.where(np.isnan(values), self.medians, values)
        return ((clean - self.means) / self.scales).astype(np.float32)

    def as_dict(self) -> dict[str, Any]:
        return {"feature_names": list(self.feature_names), "imputation": "train_median",
                "medians": self.medians.tolist(), "means": self.means.tolist(),
                "scales": self.scales.tolist(), "fitted_split": self.fitted_split}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_and_load_dataset(dataset_path: Path, metadata_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detiene el proceso si el CSV congelado no coincide con su metadata."""
    metadata = json.loads(metadata_path.read_text())
    actual = file_sha256(dataset_path)
    if actual != metadata["dataset_fingerprint"]:
        raise RuntimeError(f"Dataset Version 1 fue modificado: esperado {metadata['dataset_fingerprint']}, obtenido {actual}")
    with dataset_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["timestamp_dt"] = datetime.fromisoformat(row["timestamp"])
        row["amount"] = float(row["amount"])
        row["distance_from_home_km"] = float(row["distance_from_home_km"])
        row["is_international"] = int(row["is_international"])
        row["is_fraud"] = int(row["is_fraud"])
    rows.sort(key=lambda r: (r["timestamp_dt"], r["card_id"], r["transaction_id"]))
    assert len(rows) == metadata["n_transactions"]
    return rows, metadata


def choose_temporal_boundaries(rows: list[dict[str, Any]]) -> tuple[datetime, datetime]:
    """Elige timestamps completos cercanos a 70% y 85%, sin partir empates."""
    timestamps = [r["timestamp_dt"] for r in rows]
    first = timestamps[int(len(rows) * .70) - 1]
    second = timestamps[int(len(rows) * .85) - 1]
    assert first < second
    return first, second


def split_for_timestamp(timestamp: datetime, cutoff_train: datetime, cutoff_validation: datetime) -> str:
    if timestamp <= cutoff_train:
        return "train"
    if timestamp <= cutoff_validation:
        return "validation"
    return "test"


def _cyclical(timestamp: datetime) -> tuple[float, float, float, float]:
    hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
    return (math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * timestamp.weekday() / 7), math.cos(2 * math.pi * timestamp.weekday() / 7))


def _hours_between(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / 3600


def current_numeric(row: dict[str, Any], previous: dict[str, Any]) -> list[float]:
    return [row["amount"], row["distance_from_home_km"], row["is_international"],
            *_cyclical(row["timestamp_dt"]), _hours_between(row["timestamp_dt"], previous["timestamp_dt"])]


def historical_numeric(history: list[dict[str, Any]]) -> np.ndarray:
    """Representa eventos anteriores; el primer delta queda NaN para imputar con TRAIN."""
    result = []
    for index, row in enumerate(history):
        delta = np.nan if index == 0 else _hours_between(row["timestamp_dt"], history[index - 1]["timestamp_dt"])
        result.append([row["amount"], row["distance_from_home_km"], row["is_international"],
                       *_cyclical(row["timestamp_dt"]), delta])
    return np.asarray(result, dtype=np.float64)


def aggregate_features(target: dict[str, Any], history: list[dict[str, Any]]) -> list[float]:
    """Calcula resúmenes exclusivamente con eventos previos al target."""
    amounts = np.asarray([r["amount"] for r in history])
    distances = np.asarray([r["distance_from_home_km"] for r in history])
    mean, std = float(amounts.mean()), float(amounts.std(ddof=0))
    distance_mean = float(distances.mean())
    channel_count = sum(r["channel"] == target["channel"] for r in history)
    merchant_count = sum(r["merchant_category"] == target["merchant_category"] for r in history)
    recent: dict[int, list[dict[str, Any]]] = {}
    for hours in (1, 6, 24, 168):
        recent[hours] = [r for r in history if 0 <= _hours_between(target["timestamp_dt"], r["timestamp_dt"]) <= hours]
    day = recent[24]
    day_amounts = [r["amount"] for r in day]
    safe_mean = max(mean, 1e-6)
    safe_std = max(std, 1e-6)
    safe_distance = max(distance_mean, 1e-6)
    return [mean, std, float(amounts.max()), len(history), target["amount"] / safe_mean,
            (target["amount"] - mean) / safe_std if len(history) >= 2 and std > 1e-6 else 0.0,
            distance_mean, target["distance_from_home_km"] / safe_distance,
            channel_count / len(history), float(channel_count == 0), merchant_count / len(history),
            float(merchant_count == 0), len(recent[1]), len(recent[6]), len(day), len(recent[168]),
            float(np.mean(day_amounts)) if day_amounts else np.nan,
            float(max(day_amounts)) if day_amounts else np.nan, float(sum(day_amounts)),
            len({r["merchant_category"] for r in day}), len({r["channel"] for r in day}),
            sum(r["is_international"] for r in day), float(len(history) < 3)]


def build_raw_examples(rows: list[dict[str, Any]], cutoff_train: datetime, cutoff_validation: datetime,
                       max_history: int = MAX_HISTORY) -> dict[str, Any]:
    """Construye un universo común, agregados y secuencias previas sin transformar."""
    by_card: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transaction_lookup = {row["transaction_id"]: row for row in rows}
    for row in rows:
        by_card[row["card_id"]].append(row)
    examples, aggregates, currents, sequences, sequence_categories = [], [], [], [], []
    for card_id in sorted(by_card):
        card_rows = sorted(by_card[card_id], key=lambda r: (r["timestamp_dt"], r["transaction_id"]))
        for index in range(MIN_HISTORY, len(card_rows)):
            target = card_rows[index]
            history = card_rows[:index]
            selected = history[-max_history:]
            # transaction_id resuelve empates y define qué ocurrió antes dentro de la marca.
            assert all((r["timestamp_dt"], r["transaction_id"]) <
                       (target["timestamp_dt"], target["transaction_id"]) for r in selected)
            examples.append({"example_id": f"EX_{len(examples) + 1:09d}",
                "transaction_id": target["transaction_id"], "card_id": card_id,
                "target_timestamp": target["timestamp"], "target": target["is_fraud"],
                "split": split_for_timestamp(target["timestamp_dt"], cutoff_train, cutoff_validation),
                "fraud_type": target["fraud_type"], "hard_negative_type": target["hard_negative_type"],
                "history_length_total": len(history), "history_length_real": len(selected)})
            aggregates.append(aggregate_features(target, history))
            currents.append(current_numeric(target, history[-1]))
            sequences.append(historical_numeric(selected))
            sequence_categories.append([(r["merchant_category"], r["channel"]) for r in selected])
    return {"examples": examples, "aggregates": np.asarray(aggregates), "currents": np.asarray(currents),
            "sequences": sequences, "sequence_categories": sequence_categories,
            "current_categories": [(transaction_lookup[e["transaction_id"]]["merchant_category"],
                                    transaction_lookup[e["transaction_id"]]["channel"]) for e in examples]}


def fit_vocabularies(raw: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Aprende vocabularios únicamente de valores observados en targets TRAIN."""
    train_indices = [i for i, e in enumerate(raw["examples"]) if e["split"] == "train"]
    merchants, channels = set(), set()
    for index in train_indices:
        merchant, channel = raw["current_categories"][index]
        merchants.add(merchant); channels.add(channel)
        for merchant, channel in raw["sequence_categories"][index]:
            merchants.add(merchant); channels.add(channel)
    return {"merchant_category": {"PAD": PAD_INDEX, "UNK": UNK_INDEX,
            **{value: i + 2 for i, value in enumerate(sorted(merchants))}},
            "channel": {"PAD": PAD_INDEX, "UNK": UNK_INDEX,
            **{value: i + 2 for i, value in enumerate(sorted(channels))}}}


def transform_and_pad(raw: dict[str, Any], max_history: int = MAX_HISTORY) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Ajusta con TRAIN y produce arrays compartidos para los tres modelos."""
    split_array = np.asarray([e["split"] for e in raw["examples"]])
    train = split_array == "train"
    aggregate_scaler = Standardizer(AGGREGATE_FEATURES).fit(raw["aggregates"][train])
    current_scaler = Standardizer(CURRENT_NUMERIC_FEATURES).fit(raw["currents"][train])
    train_sequence_rows = np.concatenate([raw["sequences"][i] for i in np.flatnonzero(train)], axis=0)
    sequence_scaler = Standardizer(SEQUENCE_NUMERIC_FEATURES).fit(train_sequence_rows)
    vocabularies = fit_vocabularies(raw)
    n = len(raw["examples"])
    sequence_numeric = np.zeros((n, max_history, len(SEQUENCE_NUMERIC_FEATURES)), dtype=np.float32)
    sequence_categorical = np.zeros((n, max_history, 2), dtype=np.int16)
    sequence_mask = np.zeros((n, max_history), dtype=np.bool_)
    for index, (numeric, categories) in enumerate(zip(raw["sequences"], raw["sequence_categories"])):
        length = len(numeric); start = max_history - length
        sequence_numeric[index, start:] = sequence_scaler.transform(numeric)
        sequence_mask[index, start:] = True
        sequence_categorical[index, start:, 0] = [vocabularies["merchant_category"].get(v[0], UNK_INDEX) for v in categories]
        sequence_categorical[index, start:, 1] = [vocabularies["channel"].get(v[1], UNK_INDEX) for v in categories]
    current_categorical = np.asarray([[vocabularies["merchant_category"].get(m, UNK_INDEX),
                                       vocabularies["channel"].get(c, UNK_INDEX)]
                                      for m, c in raw["current_categories"]], dtype=np.int16)
    arrays = {"aggregate": aggregate_scaler.transform(raw["aggregates"]),
              "current_numeric": current_scaler.transform(raw["currents"]),
              "current_categorical": current_categorical, "sequence_numeric": sequence_numeric,
              "sequence_categorical": sequence_categorical, "sequence_mask": sequence_mask,
              "history_length": np.asarray([e["history_length_real"] for e in raw["examples"]], dtype=np.int16),
              "y": np.asarray([e["target"] for e in raw["examples"]], dtype=np.int8)}
    artifacts = {"aggregate_scaler": aggregate_scaler.as_dict(), "current_scaler": current_scaler.as_dict(),
                 "sequence_scaler": sequence_scaler.as_dict(), "vocabularies": vocabularies,
                 "pad_index": PAD_INDEX, "unk_index": UNK_INDEX, "fitted_split": "train"}
    return arrays, artifacts


def validate_processed(rows: list[dict[str, Any]], raw: dict[str, Any], arrays: dict[str, np.ndarray],
                       artifacts: dict[str, Any], cutoff_train: datetime, cutoff_validation: datetime) -> dict[str, Any]:
    """Ejecuta controles explícitos de temporalidad, igualdad y leakage."""
    examples = raw["examples"]
    assert not any(name in FORBIDDEN_FEATURE_NAMES for name in
                   CURRENT_NUMERIC_FEATURES + CURRENT_CATEGORICAL_FEATURES + AGGREGATE_FEATURES +
                   SEQUENCE_NUMERIC_FEATURES + SEQUENCE_CATEGORICAL_FEATURES)
    assert not any("fraud_type" in name for name in AGGREGATE_FEATURES)
    assert all(value["fitted_split"] == "train" for key, value in artifacts.items() if key.endswith("scaler"))
    assert artifacts["fitted_split"] == "train"
    train_times = [datetime.fromisoformat(e["target_timestamp"]) for e in examples if e["split"] == "train"]
    val_times = [datetime.fromisoformat(e["target_timestamp"]) for e in examples if e["split"] == "validation"]
    test_times = [datetime.fromisoformat(e["target_timestamp"]) for e in examples if e["split"] == "test"]
    assert max(train_times) < min(val_times) and max(val_times) < min(test_times)
    assert max(train_times) == cutoff_train and max(val_times) == cutoff_validation
    n = len(examples)
    assert all(array.shape[0] == n for array in arrays.values())
    assert np.array_equal(arrays["history_length"], arrays["sequence_mask"].sum(axis=1))
    assert not arrays["sequence_mask"][:, 0].all()  # existe left padding
    assert np.all(arrays["sequence_categorical"][~arrays["sequence_mask"]] == PAD_INDEX)
    assert np.all(arrays["sequence_categorical"][arrays["sequence_mask"]] >= 2)
    assert PAD_INDEX != UNK_INDEX
    lookup = {r["transaction_id"]: r for r in rows}
    card_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        card_history[row["card_id"]].append(row)
    sample_indices = np.linspace(0, n - 1, 25, dtype=int)
    aggregate_checks = []
    for i in sample_indices:
        e = examples[i]; target = lookup[e["transaction_id"]]
        prior = [r for r in card_history[e["card_id"]]
                 if (r["timestamp_dt"], r["transaction_id"]) < (target["timestamp_dt"], target["transaction_id"])]
        expected_count = sum(0 <= _hours_between(target["timestamp_dt"], r["timestamp_dt"]) <= 1 for r in prior)
        assert expected_count == raw["aggregates"][i, AGGREGATE_FEATURES.index("tx_count_1h")]
        assert np.isclose(np.mean([r["amount"] for r in prior]),
                          raw["aggregates"][i, AGGREGATE_FEATURES.index("historical_mean_amount")])
        selected = prior[-MAX_HISTORY:]
        assert selected[-1]["transaction_id"] != target["transaction_id"]
        assert selected[-1]["timestamp_dt"] <= target["timestamp_dt"]
        assert all((a["timestamp_dt"], a["transaction_id"]) < (b["timestamp_dt"], b["transaction_id"])
                   for a, b in zip(selected, selected[1:]))
        aggregate_checks.append(e["example_id"])
    return {"tests_passed": 12, "sampled_aggregate_and_sequence_checks": aggregate_checks,
            "strict_temporal_boundaries": True, "common_example_universe": True,
            "history_only_before_target": True, "train_only_fit": True,
            "padding_and_mask_valid": True, "forbidden_columns_absent": True}


def write_example_index(examples: list[dict[str, Any]], path: Path) -> None:
    fields = tuple(examples[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(examples)


def write_aggregate_features(raw: dict[str, Any], path: Path) -> None:
    fields = ("example_id", *AGGREGATE_FEATURES)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        for example, values in zip(raw["examples"], raw["aggregates"]):
            writer.writerow([example["example_id"], *values.tolist()])


def split_summary(examples: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    summary, mechanisms = {}, {name: {} for name in ("testing_cashout", "channel_takeover", "amount_anomaly")}
    for split in ("train", "validation", "test"):
        selected = [e for e in examples if e["split"] == split]
        frauds = sum(e["target"] for e in selected)
        summary[split] = {"start": min(e["target_timestamp"] for e in selected),
                          "end": max(e["target_timestamp"] for e in selected), "n": len(selected),
                          "frauds": frauds, "fraud_rate": frauds / len(selected)}
        for mechanism in mechanisms:
            mechanisms[mechanism][split] = sum(e["fraud_type"] == mechanism for e in selected)
    return summary, mechanisms


def generate_eda_figures(raw: dict[str, Any], rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    """Guarda seis gráficos descriptivos, sin predicciones ni métricas de modelo."""
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = raw["examples"]
    created = []
    monthly_all = Counter(e["target_timestamp"][:7] for e in examples)
    monthly_fraud = Counter(e["target_timestamp"][:7] for e in examples if e["target"])
    fig, ax1 = plt.subplots(figsize=(9, 5)); ax2 = ax1.twinx()
    ax1.plot(monthly_all.keys(), monthly_all.values(), marker="o", label="Todas", color="#3b82a0")
    ax2.plot(monthly_fraud.keys(), monthly_fraud.values(), marker="s", label="Fraude", color="#c75c5c")
    ax1.set(title="Transacciones objetivo y fraude por mes", xlabel="Mes", ylabel="Transacciones")
    ax2.set_ylabel("Fraudes"); fig.tight_layout(); path=output_dir/"eda_temporal_activity.png"; fig.savefig(path,dpi=140);plt.close(fig);created.append(str(path))
    mechanisms = Counter(e["fraud_type"] for e in examples if e["target"])
    fig, ax=plt.subplots(figsize=(8,5));ax.bar(mechanisms.keys(),mechanisms.values(),color="#c75c5c");ax.set(title="Fraude por mecanismo",xlabel="Mecanismo",ylabel="Transacciones");ax.tick_params(axis="x",rotation=15);fig.tight_layout();path=output_dir/"eda_fraud_mechanisms.png";fig.savefig(path,dpi=140);plt.close(fig);created.append(str(path))
    lengths=np.asarray([e["history_length_real"] for e in examples]);fig,ax=plt.subplots(figsize=(8,5));ax.hist(lengths,bins=np.arange(.5,13.6,1),color="#7251a3");ax.set(title="Eventos válidos en la secuencia",xlabel="History length (máximo 12)",ylabel="Ejemplos");fig.tight_layout();path=output_dir/"eda_history_length.png";fig.savefig(path,dpi=140);plt.close(fig);created.append(str(path))
    split_counts=Counter(e["split"] for e in examples);fig,ax=plt.subplots(figsize=(8,5));ax.bar(split_counts.keys(),split_counts.values(),color=("#5b8c5a","#d39c43","#7b6ca8"));ax.set(title="Partición temporal de targets",xlabel="Split",ylabel="Ejemplos");fig.tight_layout();path=output_dir/"eda_temporal_split.png";fig.savefig(path,dpi=140);plt.close(fig);created.append(str(path))
    lookup = {row["transaction_id"]: row for row in rows}
    amount_groups = {name: [lookup[e["transaction_id"]]["amount"] for e in examples if e["fraud_type"] == name]
                     for name in ("legitimate", "testing_cashout", "channel_takeover", "amount_anomaly")}
    fig, ax=plt.subplots(figsize=(9,5));ax.boxplot([np.log1p(amount_groups[name]) for name in amount_groups],tick_labels=list(amount_groups));ax.set(title="Monto por mecanismo (escala log1p)",xlabel="Mecanismo",ylabel="log(1 + monto Q)");ax.tick_params(axis="x",rotation=15);fig.tight_layout();path=output_dir/"eda_amount_by_mechanism.png";fig.savefig(path,dpi=140);plt.close(fig);created.append(str(path))
    channel_label = defaultdict(lambda: [0, 0])
    for example in examples:
        channel = lookup[example["transaction_id"]]["channel"]
        channel_label[channel][example["target"]] += 1
    labels=list(channel_label);legit=[channel_label[x][0] for x in labels];fraud=[channel_label[x][1] for x in labels]
    x=np.arange(len(labels));fig,ax=plt.subplots(figsize=(8,5));ax.bar(x-.18,legit,width=.36,label="Legítima");ax.bar(x+.18,fraud,width=.36,label="Fraude");ax.set_yscale("log");ax.set_xticks(x,labels);ax.set(title="Canal por etiqueta (escala log)",xlabel="Canal",ylabel="Transacciones");ax.legend();fig.tight_layout();path=output_dir/"eda_channel_by_label.png";fig.savefig(path,dpi=140);plt.close(fig);created.append(str(path))
    return created


def processed_fingerprint(paths: list[Path]) -> str:
    """Hash estable de archivos procesados principales, excluyendo timestamps informativos."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode())
        if path.suffix == ".npz":
            with np.load(path) as archive:
                for key in sorted(archive.files):
                    array = archive[key]
                    digest.update(key.encode())
                    digest.update(str(array.dtype).encode())
                    digest.update(str(array.shape).encode())
                    digest.update(array.tobytes(order="C"))
        else:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run_preprocessing(root: Path = Path("."), max_history: int = MAX_HISTORY) -> dict[str, Any]:
    generated = root / "data/generated"
    processed = root / "data/processed"
    artifacts_dir = root / "artefactos/preprocessing"
    processed.mkdir(parents=True, exist_ok=True); artifacts_dir.mkdir(parents=True, exist_ok=True)
    rows, source_metadata = verify_and_load_dataset(generated / "transactions.csv", generated / "dataset_metadata.json")
    cutoff_train, cutoff_validation = choose_temporal_boundaries(rows)
    raw = build_raw_examples(rows, cutoff_train, cutoff_validation, max_history)
    arrays, artifacts = transform_and_pad(raw, max_history)
    validation = validate_processed(rows, raw, arrays, artifacts, cutoff_train, cutoff_validation)
    summary, mechanisms = split_summary(raw["examples"])
    split_config = {"dataset_fingerprint": source_metadata["dataset_fingerprint"], "percentages": [70, 15, 15],
                    "assignment": "target timestamp; complete timestamp blocks",
                    "train_start": summary["train"]["start"], "train_end": summary["train"]["end"],
                    "validation_start": summary["validation"]["start"], "validation_end": summary["validation"]["end"],
                    "test_start": summary["test"]["start"], "test_end": summary["test"]["end"]}
    (processed / "split_config.json").write_text(json.dumps(split_config, indent=2) + "\n")
    write_example_index(raw["examples"], processed / "example_index.csv")
    write_aggregate_features(raw, processed / "aggregate_features_raw.csv")
    for split in ("train", "validation", "test"):
        indices = np.asarray([i for i, e in enumerate(raw["examples"]) if e["split"] == split])
        np.savez_compressed(processed / f"model_inputs_{split}.npz",
            example_index=indices, X_aggregate=arrays["aggregate"][indices],
            X_current_numeric=arrays["current_numeric"][indices],
            X_current_categorical=arrays["current_categorical"][indices],
            X_sequence_numeric=arrays["sequence_numeric"][indices],
            X_sequence_categorical=arrays["sequence_categorical"][indices],
            sequence_mask=arrays["sequence_mask"][indices], history_length=arrays["history_length"][indices],
            y=arrays["y"][indices])
    artifact_paths = []
    for name in ("aggregate_scaler", "current_scaler", "sequence_scaler", "vocabularies"):
        path = artifacts_dir / f"{name}.json"; path.write_text(json.dumps(artifacts[name], indent=2) + "\n"); artifact_paths.append(path)
    eda = generate_eda_figures(raw, rows, root / "figures")
    primary_paths = [processed / "split_config.json", processed / "example_index.csv",
                     processed / "aggregate_features_raw.csv", *artifact_paths,
                     *(processed / f"model_inputs_{s}.npz" for s in ("train", "validation", "test"))]
    fingerprint = processed_fingerprint(primary_paths)
    history_total = np.asarray([e["history_length_total"] for e in raw["examples"]])
    metadata = {"preprocessing_version": PREPROCESSING_VERSION,
        "dataset_source_fingerprint": source_metadata["dataset_fingerprint"],
        "processed_fingerprint": fingerprint, "min_history": MIN_HISTORY, "max_history": max_history,
        "excluded_first_transactions": len(rows) - len(raw["examples"]), "n_examples": len(raw["examples"]),
        "split_summary": summary, "fraud_by_mechanism_and_split": mechanisms,
        "aggregate_features": list(AGGREGATE_FEATURES), "current_numeric_features": list(CURRENT_NUMERIC_FEATURES),
        "current_categorical_features": list(CURRENT_CATEGORICAL_FEATURES),
        "sequence_numeric_features": list(SEQUENCE_NUMERIC_FEATURES),
        "sequence_categorical_features": list(SEQUENCE_CATEGORICAL_FEATURES),
        "sequence_shape_per_example": [max_history, len(SEQUENCE_NUMERIC_FEATURES)],
        "excluded_columns": sorted(FORBIDDEN_FEATURE_NAMES), "scalers": [str(p) for p in artifact_paths[:3]],
        "vocabularies": artifacts["vocabularies"], "fit_split": "train", "padding": "left/PAD=0/UNK=1",
        "history_distribution": {"mean_total": float(history_total.mean()), "median_total": float(np.median(history_total)),
            "min_total": int(history_total.min()), "max_total": int(history_total.max()),
            "pct_at_least_3": float((history_total >= 3).mean()), "pct_at_least_6": float((history_total >= 6).mean()),
            "pct_at_least_12": float((history_total >= 12).mean())},
        "validation": validation, "eda_figures": eda,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (processed / "processed_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_split(split: str, root: Path = Path(".")) -> dict[str, np.ndarray]:
    """Carga de forma común las entradas de A/B/C para un split congelado."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split debe ser train, validation o test")
    with np.load(root / f"data/processed/model_inputs_{split}.npz") as archive:
        return {key: archive[key] for key in archive.files}


def load_split_config(root: Path = Path(".")) -> dict[str, Any]:
    return json.loads((root / "data/processed/split_config.json").read_text())


def main() -> None:
    metadata = run_preprocessing()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
