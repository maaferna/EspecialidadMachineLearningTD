# src/data/loader.py
# -*- coding: utf-8 -*-
"""Utilidades de carga/guardado de corpus de texto (una nota por línea)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def load_corpus(path: str | Path, encoding: str = "utf-8") -> List[str]:
    """
    Lee un archivo de texto con **una nota por línea** y devuelve
    una lista de strings (sin líneas vacías y con espacios normalizados).

    Parameters
    ----------
    path : str | Path
        Ruta del archivo (ej. 'data/clinical_notes.txt').
    encoding : str
        Codificación del archivo.

    Returns
    -------
    List[str]
        Lista de documentos (notas) en bruto.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"No se encontró el corpus en {p}. "
            f"Si es la primera vez, usa 'ensure_example_corpus({repr(p)})' "
            f"o crea el archivo con una nota por línea."
        )

    docs: List[str] = []
    with p.open("r", encoding=encoding) as f:
        for line in f:
            txt = " ".join(line.strip().split())  # colapsa espacios
            if txt:
                docs.append(txt)
    return docs


def save_corpus(path: str | Path, docs: Iterable[str], encoding: str = "utf-8") -> None:
    """
    Guarda un iterable de textos al formato **una nota por línea**.

    Parameters
    ----------
    path : str | Path
        Ruta de salida.
    docs : Iterable[str]
        Textos a persistir.
    encoding : str
        Codificación del archivo.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding=encoding) as f:
        for d in docs:
            d = " ".join(str(d).strip().split())
            if d:
                f.write(d + "\n")


def ensure_example_corpus(path: str | Path, overwrite: bool = False) -> Path:
    """
    Crea un corpus de ejemplo (10 notas clínicas simuladas) si no existe.
    Útil para tests o primera corrida.

    Parameters
    ----------
    path : str | Path
        Ruta donde crear el archivo.
    overwrite : bool
        Si True, sobrescribe aunque exista.

    Returns
    -------
    Path
        Ruta final del archivo creado/existente.
    """
    p = Path(path)
    if p.exists() and not overwrite:
        return p

    example_docs = [
        "Paciente masculino, 45 años, presenta fiebre leve y congestión nasal. Se sospecha infección viral.",
        "Paciente femenina, 32 años, dolor abdominal persistente, sin fiebre, con historial de gastritis.",
        "Varón 60 años, tos seca y malestar general de 3 días. Niega disnea. Saturación 97%.",
        "Mujer 28 años, cefalea y dolor de garganta. Temperatura 37.8°C. Se indica reposo e hidratación.",
        "Hombre 50 años, hipertenso, refiere dolor torácico leve tras esfuerzo. ECG sin alteraciones agudas.",
        "Niño 8 años, otalgia derecha, fiebre 38°C. Se observan signos de otitis media aguda.",
        "Mujer 70 años, mareos ocasionales, sin vómitos. TA 130/80. Glucemia controlada.",
        "Paciente 40 años con rinitis alérgica estacional. Congestión y estornudos frecuentes.",
        "Hombre 36 años, dolor lumbar tras levantar peso. Se indica analgesia y reposo relativo.",
        "Paciente 55 años, control de diabetes tipo 2. Buen apego a tratamiento, HbA1c en rango objetivo.",
    ]
    save_corpus(p, example_docs)
    return p
