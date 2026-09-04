"""Decisiones binarias y costo económico común a todos los modelos.

La aproximación académica supone Q4,200 de daño por cada fraude no detectado
y Q180 por cada operación legítima bloqueada. No afirma que cada verdadero
positivo ahorre exactamente Q4,200 en un sistema real.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

FALSE_NEGATIVE_COST_GTQ: int = 4_200
FALSE_POSITIVE_COST_GTQ: int = 180


@dataclass(frozen=True)
class DecisionMetrics:
    threshold: float
    TP: int
    FP: int
    TN: int
    FN: int
    precision: float
    recall: float
    F1: float
    economic_cost: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def economic_cost(false_negatives: int, false_positives: int) -> int:
    """Costo total en quetzales bajo la aproximación acordada."""
    if false_negatives < 0 or false_positives < 0:
        raise ValueError("Los conteos de errores no pueden ser negativos")
    return false_negatives * FALSE_NEGATIVE_COST_GTQ + false_positives * FALSE_POSITIVE_COST_GTQ


def decision_metrics(y_true: np.ndarray, risk_score: np.ndarray, threshold: float) -> DecisionMetrics:
    """Aplica ``risk_score >= threshold`` y calcula métricas de forma segura."""
    y = np.asarray(y_true)
    score = np.asarray(risk_score, dtype=float)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score):
        raise ValueError("y_true y risk_score deben ser vectores del mismo tamaño")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true debe ser binario")
    if not np.isfinite(score).all() or not np.logical_and(score >= 0, score <= 1).all():
        raise ValueError("risk_score debe ser finito y pertenecer a [0, 1]")
    if not np.isfinite(threshold):
        raise ValueError("threshold debe ser finito")
    predicted = score >= threshold
    positive = y == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DecisionMetrics(float(threshold), tp, fp, tn, fn, precision, recall, f1,
                           economic_cost(fn, fp))


def threshold_candidates(risk_score: np.ndarray) -> np.ndarray:
    """Valores exactos observados más extremos que representan todas las decisiones."""
    score = np.asarray(risk_score, dtype=float)
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all():
        raise ValueError("Se requieren scores finitos")
    # 0 bloquea todo; nextafter(max, +inf) permite no bloquear nada.
    return np.unique(np.concatenate(([0.0], score, [np.nextafter(float(score.max()), np.inf)])))


def analyze_thresholds(y_true: np.ndarray, risk_score: np.ndarray) -> list[DecisionMetrics]:
    return [decision_metrics(y_true, risk_score, value) for value in threshold_candidates(risk_score)]


def select_economic_threshold(y_true: np.ndarray, risk_score: np.ndarray) -> tuple[DecisionMetrics, list[DecisionMetrics]]:
    """Minimiza costo; ante empate elige el threshold más alto."""
    rows = analyze_thresholds(y_true, risk_score)
    minimum = min(row.economic_cost for row in rows)
    return max((row for row in rows if row.economic_cost == minimum), key=lambda row: row.threshold), rows


def monthly_normalize(cost: float, test_days: float, days_per_month: float = 30.44) -> float:
    if test_days <= 0 or days_per_month <= 0:
        raise ValueError("Las duraciones deben ser positivas")
    return float(cost) / test_days * days_per_month
