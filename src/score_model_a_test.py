"""Adaptador de inferencia TEST para el entorno exacto de Modelo A (sklearn 1.9)."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

from .train_model_a import approved_matrix


def main() -> None:
    root = Path(sys.argv[1])
    output = Path(sys.argv[2])
    with np.load(root / "data/processed/model_inputs_test.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    payload = joblib.load(root / "artefactos/model_a/model_a.joblib")
    scores = payload["estimator"].predict_proba(approved_matrix(arrays))[:, 1]
    np.save(output, scores, allow_pickle=False)


if __name__ == "__main__":
    main()
