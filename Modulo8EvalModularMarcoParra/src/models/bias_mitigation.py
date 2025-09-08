"""Estrategias básicas de mitigación (clasificación)."""
from __future__ import annotations
from typing import Literal
import numpy as np




def threshold_adjustment(scores: np.ndarray, group: np.ndarray, target_group: str, delta: float = 0.05) -> np.ndarray:
    """Ajuste simple de umbral por grupo: desplaza las puntuaciones del grupo objetivo.
    scores: salida de decision_function o probas[:,k].
    group: vector de grupos (strings)
    target_group: grupo a compensar.
    delta: desplazamiento (positivo favorece clasificar como clase positiva si se usa 1-vs-rest)
    return: scores ajustados (misma forma que scores)
    """
    adj = scores.copy()
    adj[group == target_group] = adj[group == target_group] + delta  # Permite favorecer o desfavorecer
    return adj