from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion(cm, labels, out_path):
    """
    Genera y guarda una matriz de confusión como imagen.
    Args:
        cm: Matriz de confusión (2D array-like).
        labels: Lista de etiquetas para los ejes.
        out_path: Ruta para guardar la imagen.
    """
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), va='center', ha='center')
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()
