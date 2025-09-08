"""Descarga o genera un dataset pequeño de notas clínicas."""
from __future__ import annotations


import os
import logging
from pathlib import Path
from typing import List


import pandas as pd
import requests


from src.utils.io import ensure_dir, save_csv


logger = logging.getLogger(__name__)


DEFAULT_SYNTHETIC_NOTES: List[str] = [
"Paciente 001: Consulta por cefalea persistente y mareo leve. No refiere fiebre.",
"Paciente 002: Diarrea aguda de 24 h, malestar general y dolor abdominal leve.",
"Paciente 003: Dolor torácico opresivo, irradiado a brazo izquierdo; ECG pendiente.",
"Paciente 004: Tos seca nocturna, antecedente de asma. Saturación 96% al aire.",
"Paciente 005: Hiperglicemia en ayuno; plan de control y ajuste de dieta.",
"Paciente 006: Lumbalgia mecánica tras esfuerzo; analgesia y reposo relativo.",
"Paciente 007: Dermatitis pruriginosa en antebrazos; posible contacto con irritante.",
"Paciente 008: Rinitis alérgica estacional; indica antihistamínico de segunda generación.",
"Paciente 009: Insomnio de conciliación, escala de estrés elevada; higiene del sueño.",
"Paciente 010: Náuseas esporádicas postprandiales; descartar intolerancia alimentaria.",
]




def download_csv(url: str, dest_csv: Path, timeout: int = 20) -> pd.DataFrame:
    """Descarga un CSV plano desde URL y lo guarda en dest_csv.
    Debe contener una columna 'text'.
    """
    logger.info("Descargando dataset desde %s", url)
    ensure_dir(dest_csv.parent)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(dest_csv, "wb") as f:
        f.write(r.content)
    df = pd.read_csv(dest_csv)
    if "text" not in df.columns:
        raise ValueError("El CSV descargado debe contener una columna 'text'.")
    return df


def generate_synthetic(n: int = 10) -> pd.DataFrame:
    """Genera un DataFrame sintético con una columna 'text'."""
    notes = DEFAULT_SYNTHETIC_NOTES[:n]
    return pd.DataFrame({"text": notes})




def get_small_clinical_notes(cfg: dict) -> pd.DataFrame:
    """Obtiene un dataset pequeño de notas clínicas.


    Prioridad:
    1) cfg['dataset']['url'] si existe.
    2) ENV CLINICAL_NOTES_URL si existe.
    3) Sintético (fallback).
    """
    ds_cfg = cfg.get("dataset", {})
    url = (ds_cfg.get("url") or os.getenv("CLINICAL_NOTES_URL") or "").strip()
    out_csv = Path(ds_cfg.get("out_csv", "data/raw/clinical_notes.csv"))


    if url:
        try:
            df = download_csv(url, out_csv)
            logger.info("Dataset descargado: %s (%d filas)", out_csv, len(df))
            return df
        except Exception as e:
            logger.warning("Fallo descarga (%s). Usando sintético.", e)


    df = generate_synthetic(n=10)
    save_csv(df, out_csv)
    logger.info("Dataset sintético generado en %s (%d filas)", out_csv, len(df))
    return df