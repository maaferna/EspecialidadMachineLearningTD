# src/similarity/metrics.py
from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim_matrix(X) -> np.ndarray:
    """Matriz de similitud coseno entre documentos (d x d)."""
    return cosine_similarity(X)
