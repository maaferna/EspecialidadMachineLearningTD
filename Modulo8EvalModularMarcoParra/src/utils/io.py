"""Utilidades de E/S."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd




def ensure_dir(path: str | Path) -> Path:
    """Crea la carpeta si no existe y devuelve el Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict, path: str | Path) -> None:
    """Guarda un diccionario como JSON.
    Parameters
    ----------
    obj : dict
        Diccionario a guardar.
    path : str | Path
        Ruta de salida.
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Guarda un DataFrame como CSV."""
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=index)