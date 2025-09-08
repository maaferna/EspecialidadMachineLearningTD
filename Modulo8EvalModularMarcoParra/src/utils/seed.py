"""Semillas y determinismo básico."""
from __future__ import annotations
import os, random, numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Fija semillas para reproducibilidad básica.
    No garantiza reproducibilidad total en GPU.
    Parameters
    ----------
    seed : int
        Valor de la semilla.
    Returns
    -------
    None
    -------
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)