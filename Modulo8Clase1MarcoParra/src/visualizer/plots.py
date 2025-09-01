# src/visualize/plots_nlp.py
from __future__ import annotations
from typing import List, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_similarity_heatmap(sim_matrix, out_path=None):
    """
    Dibuja un heatmap de similaridad entre documentos.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="Blues", square=True, cbar=True)
    plt.title("Matriz de similaridad (coseno)")
    plt.xlabel("Documento")
    plt.ylabel("Documento")

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
        print("Heatmap guardado en:", out_path)
    else:
        plt.show()
    plt.close()



def plot_top_terms_bars(top_terms, title, path):
    """Bar chart de términos (TF-IDF) para un documento."""

    terms, scores = zip(*top_terms)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=scores, y=terms, orient="h")
    plt.title(title)
    plt.xlabel("TF-IDF")
    plt.ylabel("Término")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print("Top términos guardados en:", path)


