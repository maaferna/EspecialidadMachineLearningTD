"""Carga YAML de configuración."""
from __future__ import annotations
from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    """Carga archivo YAML desde `path` y devuelve un diccionario."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)