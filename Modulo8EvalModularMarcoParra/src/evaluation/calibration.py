"""Curva de confiabilidad para modelos con predict_proba/decision_function."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from src.utils.io import ensure_dir




def reliability_curve(probs, y_true, out_path: str | Path) -> None:
    """Genera y guarda la curva de confiabilidad (reliability curve).
    Parameters
    ----------
    probs : np.ndarray
        Probabilidades o scores del modelo (n_samples,) o (n_samples, n_classes).
    y_true : np.ndarray
        Etiquetas verdaderas (0/1).
    out_path : str | Path
        Ruta de salida para la figura (PNG).
    Returns
    -------
    None"""
    ensure_dir(Path(out_path).parent)
    # Usa la probabilidad de la clase mayor riesgo (severo) si probs es (n_samples, n_classes)
    if probs.ndim == 2:
    # Tomamos la última columna por convenio (orden de label encoder)
        conf = probs.max(axis=1)
    else:
        conf = probs
        frac_pos, mean_pred = calibration_curve(y_true=(y_true), y_prob=conf, n_bins=10, strategy="uniform")
        plt.figure(figsize=(4.5,4))
        plt.plot([0,1],[0,1], linestyle="--")
        plt.plot(mean_pred, frac_pos, marker="o")
        plt.xlabel("Confianza promedio")
        plt.ylabel("Fracción positiva")
        plt.title("Curva de confiabilidad")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()