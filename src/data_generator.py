"""Generador reproducible de transacciones sintéticas (Dataset Version 1).

Crea perfiles, actividad legítima, casos difíciles y tres fraudes; después
valida y guarda un CSV estable. No construye features, splits ni modelos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import RANDOM_SEED

GENERATOR_VERSION = "1.0.0"
DATASET_VERSION = "1"
COLUMNS = ("transaction_id", "card_id", "timestamp", "amount", "merchant_category", "channel",
           "distance_from_home_km", "is_international", "is_fraud", "fraud_type", "fraud_stage",
           "customer_profile", "hard_negative_type")

# Los IDs se conservan para trazabilidad/secuencias, pero tampoco son features.
METADATA_ONLY_COLUMNS = ("transaction_id", "card_id", "is_fraud", "fraud_type", "fraud_stage",
                         "customer_profile", "hard_negative_type")
MODEL_FEATURE_CANDIDATES = ("timestamp", "amount", "merchant_category", "channel",
                            "distance_from_home_km", "is_international")
PROFILES = ("regular", "online", "high_spend", "variable", "traveler")
PROFILE_PROBABILITIES = (0.38, 0.20, 0.14, 0.16, 0.12)
CATEGORIES = ("grocery", "restaurant", "fuel", "retail", "electronics", "travel",
              "entertainment", "pharmacy", "services", "other")
CHANNELS = ("POS", "ONLINE", "ATM", "CONTACTLESS")
FRAUD_TYPES = ("testing_cashout", "channel_takeover", "amount_anomaly")
HARD_NEGATIVE_TYPES = ("legitimate_trip", "legitimate_large_purchase",
                       "legitimate_shopping_spree", "legitimate_micro_purchases")


@dataclass(frozen=True)
class GeneratorConfig:
    """Parámetros principales, sin exponer cada detalle interno."""

    random_seed: int = RANDOM_SEED
    start_date: str = "2025-01-01"
    n_days: int = 180
    n_cards: int = 2_800
    mean_transactions_per_card: float = 34.0
    fraud_episode_probability: float = 0.18
    hard_negative_probability: float = 0.22
    profile_probabilities: tuple[float, ...] = PROFILE_PROBABILITIES

    @classmethod
    def small(cls, random_seed: int = RANDOM_SEED) -> "GeneratorConfig":
        """Configuración rápida de alrededor de diez mil operaciones."""
        return cls(random_seed=random_seed, n_days=90, n_cards=400,
                   mean_transactions_per_card=23.0, fraud_episode_probability=0.16,
                   hard_negative_probability=0.25)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / values.sum()


def create_card_profiles(config: GeneratorConfig, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Crea preferencias latentes individuales usadas solo para simular."""
    weights = {
        "regular": ([.36, .17, .12, .11, .04, .03, .06, .06, .04, .01], [.57, .08, .05, .30]),
        "online": ([.16, .13, .05, .22, .10, .08, .10, .04, .09, .03], [.20, .65, .02, .13]),
        "high_spend": ([.12, .18, .06, .19, .13, .13, .07, .03, .07, .02], [.48, .34, .06, .12]),
        "variable": ([.15, .13, .08, .15, .10, .10, .09, .06, .09, .05], [.36, .31, .12, .21]),
        "traveler": ([.10, .19, .04, .13, .08, .25, .07, .03, .08, .03], [.36, .40, .09, .15]),
    }
    medians = {"regular": 115, "online": 150, "high_spend": 620, "variable": 230, "traveler": 360}
    sigmas = {"regular": .58, "online": .68, "high_spend": .72, "variable": 1., "traveler": .85}
    frequency = {"regular": .90, "online": 1.10, "high_spend": .80, "variable": 1., "traveler": 1.05}
    distances = {"regular": 7, "online": 5, "high_spend": 12, "variable": 25, "traveler": 90}
    kinds = rng.choice(PROFILES, size=config.n_cards, p=config.profile_probabilities)
    profiles = []
    for index, kind_value in enumerate(kinds):
        kind = str(kind_value)
        category_base, channel_base = weights[kind]
        profiles.append({
            "card_id": f"CARD_{index + 1:05d}", "profile": kind,
            "typical_amount": float(medians[kind] * rng.lognormal(0, .28)),
            "amount_sigma": sigmas[kind] * float(rng.uniform(.85, 1.15)),
            "transaction_rate": config.mean_transactions_per_card * frequency[kind] * float(rng.uniform(.72, 1.28)),
            "preferred_hour": float(np.clip(rng.normal(18 if kind == "online" else 15, 2.6), 8, 21)),
            "hour_spread": 3.2 if kind == "regular" else 4.8,
            "category_weights": _normalize(rng.dirichlet(np.asarray(category_base) * 28 + .3)),
            "channel_weights": _normalize(rng.dirichlet(np.asarray(channel_base) * 35 + .3)),
            "distance_scale": distances[kind],
        })
    return profiles


def _timestamp(start: datetime, day: int, hour: float, rng: np.random.Generator) -> datetime:
    seconds = int(np.clip(hour * 3600 + rng.normal(0, 22 * 60), 0, 86_399))
    return start + timedelta(days=day, seconds=seconds)


def _event(profile: dict[str, Any], timestamp: datetime, amount: float, category: str, channel: str,
           distance: float, international: bool, *, fraud_type: str = "legitimate",
           stage: str = "none", hard_negative: str = "none") -> dict[str, Any]:
    return {"card_id": profile["card_id"], "timestamp": timestamp, "amount": round(max(.5, amount), 2),
            "merchant_category": category, "channel": channel,
            "distance_from_home_km": round(max(0, distance), 2), "is_international": int(international),
            "is_fraud": int(fraud_type != "legitimate"), "fraud_type": fraud_type,
            "fraud_stage": stage, "customer_profile": profile["profile"],
            "hard_negative_type": hard_negative}


def _legitimate_event(profile: dict[str, Any], timestamp: datetime, rng: np.random.Generator) -> dict[str, Any]:
    category = str(rng.choice(CATEGORIES, p=profile["category_weights"]))
    channel = str(rng.choice(CHANNELS, p=profile["channel_weights"]))
    factor = {"grocery": .75, "restaurant": .60, "fuel": .62, "retail": 1.05, "electronics": 2.4,
              "travel": 2.8, "entertainment": .72, "pharmacy": .55, "services": 1.15, "other": .90}[category]
    amount = profile["typical_amount"] * factor * rng.lognormal(0, profile["amount_sigma"])
    p_international = .12 if profile["profile"] == "traveler" else (.035 if channel == "ONLINE" else .008)
    international = bool(rng.random() < p_international)
    distance = rng.gamma(1.7, profile["distance_scale"]) * (.25 if channel == "ONLINE" else 1)
    if international:
        distance += rng.uniform(350, 5_000)
    return _event(profile, timestamp, amount, category, channel, distance, international)


def generate_legitimate_transactions(config: GeneratorConfig, profiles: list[dict[str, Any]],
                                     rng: np.random.Generator) -> list[dict[str, Any]]:
    """Genera actividad base no uniforme por tarjeta, horario y perfil."""
    start = datetime.fromisoformat(config.start_date)
    events = []
    for profile in profiles:
        count = max(8, int(rng.poisson(profile["transaction_rate"])))
        for day in rng.integers(0, config.n_days, size=count):
            hour = float(rng.normal(profile["preferred_hour"], profile["hour_spread"]) % 24)
            events.append(_legitimate_event(profile, _timestamp(start, int(day), hour, rng), rng))
    return events


def _anchor(config: GeneratorConfig, rng: np.random.Generator) -> datetime:
    start = datetime.fromisoformat(config.start_date)
    day = int(rng.integers(4, config.n_days - 4))
    return _timestamp(start, day, float(np.clip(rng.normal(15, 4.5), 1, 23)), rng)


def inject_hard_negatives(events: list[dict[str, Any]], profiles: list[dict[str, Any]],
                          config: GeneratorConfig, rng: np.random.Generator) -> Counter[str]:
    """Añade episodios legítimos parecidos a reglas antifraude simples."""
    counts: Counter[str] = Counter()
    for profile in profiles:
        if rng.random() >= config.hard_negative_probability:
            continue
        kind = str(rng.choice(HARD_NEGATIVE_TYPES))
        anchor = _anchor(config, rng)
        if kind == "legitimate_trip":
            n = int(rng.integers(2, 5))
            for step in range(n):
                events.append(_event(profile, anchor + timedelta(hours=step * float(rng.uniform(2, 8))),
                    profile["typical_amount"] * rng.lognormal(.5, .65), str(rng.choice(("travel", "restaurant", "retail"))),
                    str(rng.choice(("POS", "ONLINE", "ATM"))), rng.uniform(400, 5_500), True, hard_negative=kind))
                counts[kind] += 1
        elif kind == "legitimate_large_purchase":
            events.append(_event(profile, anchor, profile["typical_amount"] * rng.uniform(4, 9),
                str(rng.choice(("electronics", "travel", "retail"))), str(rng.choice(("POS", "ONLINE"))),
                rng.gamma(2, profile["distance_scale"]), bool(rng.random() < .12), hard_negative=kind))
            counts[kind] += 1
        elif kind == "legitimate_shopping_spree":
            n = int(rng.integers(3, 7))
            for step in range(n):
                events.append(_event(profile, anchor + timedelta(minutes=step * int(rng.integers(8, 35))),
                    profile["typical_amount"] * rng.lognormal(-.1, .55), str(rng.choice(("retail", "restaurant", "entertainment"))),
                    str(rng.choice(("POS", "CONTACTLESS", "ONLINE"))), rng.gamma(1.8, 12), False, hard_negative=kind))
                counts[kind] += 1
        else:
            n = int(rng.integers(3, 7))
            for step in range(n):
                events.append(_event(profile, anchor + timedelta(minutes=step * int(rng.integers(3, 16))),
                    rng.uniform(2, 24), str(rng.choice(("services", "entertainment", "other"))),
                    str(rng.choice(("ONLINE", "CONTACTLESS"))), rng.uniform(0, 8), False, hard_negative=kind))
                counts[kind] += 1
    return counts


def inject_testing_cashout(events: list[dict[str, Any]], profile: dict[str, Any],
                           config: GeneratorConfig, rng: np.random.Generator) -> None:
    """Inyecta pequeñas pruebas seguidas de un cashout plausible."""
    anchor = _anchor(config, rng)
    n_probes = int(rng.integers(2, 6))
    channel = str(rng.choice(("ONLINE", "CONTACTLESS")))
    elapsed = 0
    for _ in range(n_probes):
        events.append(_event(profile, anchor + timedelta(minutes=elapsed),
            rng.uniform(3, min(32, profile["typical_amount"] * .25 + 5)), "services", channel,
            rng.uniform(0, 25), False, fraud_type="testing_cashout", stage="probe"))
        elapsed += int(rng.integers(3, 11))
    # El monto final se solapa con compras legítimas; la secuencia es la señal fuerte.
    events.append(_event(profile, anchor + timedelta(minutes=elapsed + int(rng.integers(6, 15))),
        profile["typical_amount"] * rng.lognormal(.55, .55), str(rng.choice(("retail", "electronics", "other"))),
        str(rng.choice(("ATM", "ONLINE", "POS"), p=(.45, .35, .20))), rng.uniform(3, 180),
        bool(rng.random() < .18), fraud_type="testing_cashout", stage="cashout"))


def inject_channel_takeover(events: list[dict[str, Any]], profile: dict[str, Any],
                            config: GeneratorConfig, rng: np.random.Generator) -> None:
    """Inyecta una transición corta hacia canales poco habituales."""
    anchor = _anchor(config, rng)
    rare = [CHANNELS[int(i)] for i in np.argsort(profile["channel_weights"])[:2]]
    sequence = [rare[0], rare[1], str(rng.choice(CHANNELS))]
    elapsed = 0
    for step, channel in enumerate(sequence):
        international = bool(rng.random() < .55)
        distance = rng.uniform(150, 4_500) if international or rng.random() < .7 else rng.uniform(5, 120)
        events.append(_event(profile, anchor + timedelta(minutes=elapsed),
            profile["typical_amount"] * rng.lognormal(-.05, .6), str(rng.choice(CATEGORIES)), channel,
            distance, international, fraud_type="channel_takeover",
            stage=("entry", "transition", "completion")[step]))
        elapsed += int(rng.integers(12, 46))


def inject_amount_anomaly(events: list[dict[str, Any]], profile: dict[str, Any],
                          config: GeneratorConfig, rng: np.random.Generator) -> None:
    """Inyecta una desviación de monto sin requerir una secuencia fija."""
    events.append(_event(profile, _anchor(config, rng), profile["typical_amount"] * rng.uniform(5.5, 10),
        str(rng.choice(("electronics", "travel", "retail", "services"))), str(rng.choice(CHANNELS)),
        rng.gamma(2, profile["distance_scale"]), bool(rng.random() < .18),
        fraud_type="amount_anomaly", stage="anomaly"))


def inject_fraud(events: list[dict[str, Any]], profiles: list[dict[str, Any]],
                 config: GeneratorConfig, rng: np.random.Generator) -> None:
    """Selecciona episodios sin conocer futuras fronteras temporales."""
    injectors = (inject_testing_cashout, inject_channel_takeover, inject_amount_anomaly)
    for profile in profiles:
        if rng.random() < config.fraud_episode_probability:
            injectors[int(rng.choice(3, p=(.45, .35, .20)))](events, profile, config, rng)


def generate_transactions(config: GeneratorConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Genera y ordena la línea temporal completa, todavía sin splits."""
    rng = np.random.default_rng(config.random_seed)
    profiles = create_card_profiles(config, rng)
    events = generate_legitimate_transactions(config, profiles, rng)
    hard_negatives = inject_hard_negatives(events, profiles, config, rng)
    inject_fraud(events, profiles, config, rng)
    events.sort(key=lambda row: (row["timestamp"], row["card_id"], row["amount"], row["channel"]))
    for index, event in enumerate(events, 1):
        event["transaction_id"] = f"TX_{index:09d}"
        event["timestamp"] = event["timestamp"].isoformat(timespec="seconds")
    return events, dict(sorted(hard_negatives.items()))


def stable_csv_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    """Serializa igual que el archivo para obtener un hash reproducible."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def dataset_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(stable_csv_bytes(rows)).hexdigest()


def validate_dataset(rows: list[dict[str, Any]], config: GeneratorConfig) -> dict[str, Any]:
    """Comprueba integridad, diversidad, cobertura y shortcuts directos."""
    assert rows
    ids = [r["transaction_id"] for r in rows]
    assert len(ids) == len(set(ids)), "transaction_id no es único"
    assert all(float(r["amount"]) > 0 for r in rows)
    timestamps = [datetime.fromisoformat(str(r["timestamp"])) for r in rows]
    assert timestamps == sorted(timestamps)
    assert all(r["is_fraud"] in (0, 1) for r in rows)
    assert not set(METADATA_ONLY_COLUMNS) & set(MODEL_FEATURE_CANDIDATES)
    required = {"transaction_id", "card_id", "is_fraud", "fraud_type", "fraud_stage", "hard_negative_type"}
    assert required <= set(METADATA_ONLY_COLUMNS)
    assert all("fraud" not in r["transaction_id"].lower() and "fraud" not in r["card_id"].lower() for r in rows)
    fraud_rows = [r for r in rows if r["is_fraud"]]
    fraud_types = Counter(r["fraud_type"] for r in fraud_rows)
    assert set(fraud_types) == set(FRAUD_TYPES)
    # El preset pequeño acepta cinco casos; el mínimo crece con el dataset.
    minimum = max(5, int(len(rows) * .0005))
    assert min(fraud_types.values()) >= minimum
    rate = len(fraud_rows) / len(rows)
    assert .01 <= rate <= .03, f"Prevalencia fuera de rango: {rate:.4f}"
    assert any(r["hard_negative_type"] != "none" for r in rows)
    per_card: dict[str, list[datetime]] = defaultdict(list)
    for row, stamp in zip(rows, timestamps):
        per_card[row["card_id"]].append(stamp)
    assert all(values == sorted(values) for values in per_card.values())
    start, end = min(timestamps), max(timestamps)
    span = max((end - start).total_seconds(), 1)
    thirds = [0, 0, 0]
    for row, stamp in zip(rows, timestamps):
        if row["is_fraud"]:
            thirds[min(2, int(3 * (stamp - start).total_seconds() / span))] += 1
    assert all(value >= minimum for value in thirds)
    shortcut_rates = {}
    for column in ("channel", "merchant_category", "is_international"):
        totals = Counter(str(r[column]) for r in rows)
        positives = Counter(str(r[column]) for r in fraud_rows)
        rates = {value: positives[value] / count for value, count in totals.items()}
        assert max(rates.values()) < .25, f"Shortcut categórico en {column}"
        shortcut_rates[column] = dict(sorted(rates.items()))
    hour_totals = Counter(r["timestamp"][11:13] for r in rows)
    hour_positives = Counter(r["timestamp"][11:13] for r in fraud_rows)
    hour_rates = {hour: hour_positives[hour] / total for hour, total in hour_totals.items()}
    assert len(hour_positives) >= 18, "El fraude está concentrado en muy pocas horas"
    assert max(hour_rates.values()) < .10, "Una hora revela demasiado la etiqueta"
    legitimate_amounts = np.array([r["amount"] for r in rows if not r["is_fraud"]])
    fraudulent_amounts = np.array([r["amount"] for r in fraud_rows])
    assert fraudulent_amounts.min() < np.quantile(legitimate_amounts, .90)
    assert legitimate_amounts.max() > np.quantile(fraudulent_amounts, .90)
    fraud_card_share = len({r["card_id"] for r in fraud_rows}) / len(per_card)
    assert fraud_card_share >= .08
    fraud_cards = {r["card_id"] for r in fraud_rows}
    legitimate_cards = {r["card_id"] for r in rows if not r["is_fraud"]}
    assert fraud_cards <= legitimate_cards
    return {"fraud_rate": rate, "fraud_counts": dict(sorted(fraud_types.items())),
            "fraud_counts_by_period_third": thirds, "fraud_card_share": fraud_card_share,
            "category_value_fraud_rates": shortcut_rates,
            "max_hourly_fraud_rate": max(hour_rates.values()), "hours_with_fraud": len(hour_positives),
            "all_fraud_cards_also_have_legitimate_activity": True, "amount_ranges_overlap": True}


def build_metadata(rows: list[dict[str, Any]], config: GeneratorConfig, hard_negatives: dict[str, int],
                   validation: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    fraud_counts = Counter(r["fraud_type"] for r in rows if r["is_fraud"])
    card_counts = Counter(r["card_id"] for r in rows)
    n_fraud = sum(fraud_counts.values())
    return {"dataset_version": DATASET_VERSION, "generator_version": GENERATOR_VERSION,
            "seed": config.random_seed, "start_date": rows[0]["timestamp"], "end_date": rows[-1]["timestamp"],
            "n_cards": len(card_counts), "n_transactions": len(rows), "n_legitimate": len(rows) - n_fraud,
            "n_fraud": n_fraud, "fraud_rate": n_fraud / len(rows),
            "fraud_counts_by_type": dict(sorted(fraud_counts.items())), "hard_negative_counts": hard_negatives,
            "history_length_per_card": {"min": min(card_counts.values()),
                "median": float(np.median(list(card_counts.values()))),
                "mean": float(np.mean(list(card_counts.values()))), "max": max(card_counts.values())},
            "metadata_only_columns": list(METADATA_ONLY_COLUMNS),
            "model_feature_candidates": list(MODEL_FEATURE_CANDIDATES),
            "fingerprint_algorithm": "SHA-256 over exact UTF-8 CSV bytes", "dataset_fingerprint": fingerprint,
            "validation_summary": validation,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def save_dataset(rows: list[dict[str, Any]], config: GeneratorConfig, output_dir: Path,
                 hard_negatives: dict[str, int], validation: dict[str, Any]) -> dict[str, Any]:
    """Guarda CSV, configuración y metadata; la hora no entra al hash."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_bytes = stable_csv_bytes(rows)
    fingerprint = hashlib.sha256(csv_bytes).hexdigest()
    (output_dir / "transactions.csv").write_bytes(csv_bytes)
    (output_dir / "generator_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    metadata = build_metadata(rows, config, hard_negatives, validation, fingerprint)
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def show_fraud_episode(rows: list[dict[str, Any]], fraud_type: str = "testing_cashout",
                       previous_events: int = 4) -> list[dict[str, Any]]:
    """Devuelve un episodio real junto con actividad anterior de la tarjeta."""
    target = next((r for r in rows if r["fraud_type"] == fraud_type and r["fraud_stage"] == "cashout"), None)
    target = target or next(r for r in rows if r["fraud_type"] == fraud_type)
    card_rows = [r for r in rows if r["card_id"] == target["card_id"]]
    index = card_rows.index(target)
    episode_start = index
    while episode_start and card_rows[episode_start - 1]["fraud_type"] == fraud_type:
        episode_start -= 1
    fields = ("timestamp", "card_id", "amount", "channel", "merchant_category", "is_fraud", "fraud_type", "fraud_stage")
    return [{field: row[field] for field in fields}
            for row in card_rows[max(0, episode_start - previous_events):index + 1]]


def descriptive_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Resume propiedades del dataset, no desempeño predictivo."""
    amounts = np.asarray([r["amount"] for r in rows])
    return {"transactions": len(rows), "cards": len({r["card_id"] for r in rows}),
            "period": [rows[0]["timestamp"], rows[-1]["timestamp"]],
            "fraud_rate": sum(r["is_fraud"] for r in rows) / len(rows),
            "fraud_by_type": dict(Counter(r["fraud_type"] for r in rows if r["is_fraud"])),
            "transactions_by_channel": dict(Counter(r["channel"] for r in rows)),
            "transactions_by_category": dict(Counter(r["merchant_category"] for r in rows)),
            "amount_mean": float(amounts.mean()), "amount_median": float(np.median(amounts)),
            "hard_negatives": dict(Counter(r["hard_negative_type"] for r in rows
                                            if r["hard_negative_type"] != "none"))}


def generate_sanity_figures(rows: list[dict[str, Any]], output_dir: Path = Path("figures")) -> list[Path]:
    """Crea cuatro gráficos descriptivos sin realizar análisis predictivo."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    amounts = np.asarray([r["amount"] for r in rows])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(np.clip(amounts, 0, np.quantile(amounts, .99)), bins=55, color="#3b82a0")
    axis.set(title="Distribución de montos (recortada al percentil 99)", xlabel="Monto (Q)", ylabel="Transacciones")
    path = output_dir / "amount_distribution.png"; figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure); created.append(path)

    fraud_counts = Counter(r["fraud_type"] for r in rows if r["is_fraud"])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(fraud_counts.keys(), fraud_counts.values(), color="#c75c5c")
    axis.set(title="Transacciones fraudulentas por mecanismo", xlabel="Mecanismo", ylabel="Transacciones")
    axis.tick_params(axis="x", rotation=15)
    path = output_dir / "fraud_by_type.png"; figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure); created.append(path)

    channel_counts = Counter(r["channel"] for r in rows)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(channel_counts.keys(), channel_counts.values(), color="#5b8c5a")
    axis.set(title="Transacciones por canal", xlabel="Canal", ylabel="Transacciones")
    path = output_dir / "transactions_by_channel.png"; figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure); created.append(path)

    monthly = Counter(str(r["timestamp"])[:7] for r in rows)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(monthly.keys(), monthly.values(), marker="o", color="#7251a3")
    axis.set(title="Actividad a través del tiempo", xlabel="Mes", ylabel="Transacciones")
    path = output_dir / "transactions_over_time.png"; figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure); created.append(path)
    return created


def run_reproducibility_test() -> dict[str, Any]:
    """Comprueba dos ejecuciones con seed 42 y otra con seed distinto."""
    config = GeneratorConfig.small(42)
    hashes = [dataset_fingerprint(generate_transactions(candidate)[0]) for candidate in
              (config, config, replace(config, random_seed=43))]
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]
    return {"seed_42_run_1": hashes[0], "seed_42_run_2": hashes[1], "seed_43": hashes[2],
            "same_seed_equal": True, "different_seed_different": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    args = parser.parse_args()
    config = GeneratorConfig.small() if args.small else GeneratorConfig()
    rows, hard_negatives = generate_transactions(config)
    validation = validate_dataset(rows, config)
    metadata = save_dataset(rows, config, args.output_dir, hard_negatives, validation)
    figures = generate_sanity_figures(rows)
    print(json.dumps({"metadata": metadata, "statistics": descriptive_statistics(rows),
                      "example": show_fraud_episode(rows),
                      "figures": [str(path) for path in figures]}, indent=2))


if __name__ == "__main__":
    main()
