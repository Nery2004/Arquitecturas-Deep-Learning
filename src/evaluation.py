"""Métricas comunes para los modelos del proyecto."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

PRIMARY_METRIC_NAME = "AUC-PR / Average Precision (AP)"


def average_precision(y_true: np.ndarray, risk_score: np.ndarray) -> float:
    """Implementación congelada que usarán A, B, C y falsificaciones."""
    return float(average_precision_score(y_true, risk_score))


def pr_curve(y_true: np.ndarray, risk_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curva completa, sin elegir un threshold operativo."""
    return precision_recall_curve(y_true, risk_score)
