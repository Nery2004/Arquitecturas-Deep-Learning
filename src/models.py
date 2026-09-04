"""Definiciones de modelos; B y C permanecen pendientes."""

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier


def create_model_a(parameters: dict[str, Any]) -> HistGradientBoostingClassifier:
    """Crea el candidato no lineal sin early stopping interno aleatorio."""
    return HistGradientBoostingClassifier(**parameters, early_stopping=False,
                                          categorical_features=[8, 9])


# Las arquitecturas secuencial B e híbrida C no se implementan en esta etapa.
