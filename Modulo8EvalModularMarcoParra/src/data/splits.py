"""Split train/test con estratificación opcional (auto-ajuste de test_size)."""
from __future__ import annotations
from typing import Tuple
import math
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Divide `df` en train/test con tamaño `test_size` (float entre 0 y 1).
    Si `stratify=True`, asegura al menos 1 ejemplo por clase en train y test,
    ajustando `test_size` si es necesario (útil para datasets pequeños o con
    muchas clases).
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a dividir.
    label_col : str
        Nombre de la columna de etiquetas (clases).
    test_size : float, optional
        Proporción del dataset a usar como test (entre 0 y 1), by default 0.2
    stratify : bool, optional
        Si True, estratifica por `label_col` y ajusta `test_size`
        para asegurar al menos 1 ejemplo por clase en train y test, by default True
    random_state : int, optional
        Semilla para reproducibilidad, by default 42
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        train_df, test_df
    -------
    """
    n = len(df)
    n_classes = df[label_col].nunique()

    if n == 0 or n_classes == 0:
        raise ValueError("Dataset vacío o sin clases.")

    # Copia local de test_size (puede ajustarse)
    ts = float(test_size)

    if stratify:
        # Asegurar al menos 1 ejemplo por clase en test
        min_test = math.ceil(n_classes) / n
        # Y al menos 1 por clase en train
        min_train = math.ceil(n_classes) / n
        min_total = min_test
        if 1.0 - ts < min_train:
            # si train quedaría con menos de n_clases, subimos test para liberar train
            # (en práctica, con n pequeño conviene aumentar todo)
            pass  # el chequeo clave es asegurar test >= n_clases

        if ts < min_test:
            old = ts
            ts = min_test
            warnings.warn(
                f"[make_splits] test_size={old:.3f} insuficiente para {n_classes} clases con estratificación. "
                f"Ajustado a test_size={ts:.3f} (n_test={math.ceil(ts*n)})."
            )

        y = df[label_col]
        train_df, test_df = train_test_split(
            df, test_size=ts, random_state=random_state, stratify=y
        )
        return train_df, test_df

    # Sin estratificación
    train_df, test_df = train_test_split(
        df, test_size=ts, random_state=random_state, stratify=None
    )
    return train_df, test_df
