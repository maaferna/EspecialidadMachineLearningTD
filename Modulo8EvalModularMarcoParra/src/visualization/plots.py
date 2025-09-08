"""Gráficos utilitarios (confusión, curvas)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from src.utils.io import ensure_dir




def plot_confusion(cm: np.ndarray, labels: list[str], out_path: str, title: str | None = None):
    """Grafica y guarda la matriz de confusión `cm` con etiquetas `labels` en `out_path`.
    Parameters
    ----------
    cm : np.ndarray
        Matriz de confusión (2D, cuadrada).
    labels : list[str]
        Lista de etiquetas (nombres de clases).
    out_path : str | Path
        Ruta de salida para guardar la figura (PNG).
    Returns
    -------
    None
    -------
    """


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()