"""Utilidades de entrada/salida y configuración."""
from __future__ import annotations


import json
import logging
from pathlib import Path
from typing import Iterable, List


import pandas as pd
import yaml




logger = logging.getLogger(__name__)




def ensure_dir(path: str | Path) -> Path:
    """Crea un directorio si no existe y devuelve su Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p




def load_yaml(path: str | Path) -> dict:
    """Carga YAML en un dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)




def save_json(obj: dict, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)




def save_lines(lines: Iterable[str], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")




def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)




def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=index)