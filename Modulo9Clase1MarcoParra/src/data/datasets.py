from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple

import sys
from pathlib import Path

# Añade el root del proyecto al sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_dataset(cfg) -> Tuple[pd.DataFrame, str, str]:
    """
    Carga un dataset basado en la configuración dada.
    Soporta fuentes: 'huggingface', 'csv', 'synthetic'.
    Devuelve un DataFrame y los nombres de las columnas de texto y etiqueta.
    Args:
        cfg: Configuración con detalles del dataset.
    Returns:
        df: DataFrame con los datos.
        text_col: Nombre de la columna de texto.
        label_col: Nombre de la columna de etiqueta.
    Raises:
        ValueError: Si la fuente del dataset no es soportada.
        FileNotFoundError: Si el archivo CSV no existe.
    
    """
    src = cfg["dataset"]["source"]
    text_col = cfg["dataset"]["text_col"]
    label_col = cfg["dataset"]["label_col"]

    if src == "huggingface":
        from datasets import load_dataset
        ds = load_dataset(cfg["dataset"]["hf_name"])
        train = ds["train"].to_pandas()
        test  = ds["test"].to_pandas()
        # opcional: subset para rapidez
        mtr = cfg["dataset"].get("max_samples_train")
        mts = cfg["dataset"].get("max_samples_test")
        if mtr: train = train.sample(n=min(mtr, len(train)), random_state=42)
        if mts: test  = test.sample(n=min(mts, len(test)),  random_state=42)
        df = pd.concat([train, test], ignore_index=True)
        df = df[[text_col, label_col]].dropna()
        return df, text_col, label_col

    elif src == "csv":
        path = Path(cfg["dataset"]["csv_path"])
        if not path.exists():
            raise FileNotFoundError(f"No existe CSV en {path}")
        df = pd.read_csv(path)
        if not {text_col, label_col}.issubset(df.columns):
            raise ValueError(f"CSV debe tener columnas {text_col}, {label_col}")
        return df, text_col, label_col

    elif src == "synthetic":
        n = int(cfg["dataset"].get("simulate_min", 200))
        np.random.seed(42)
        pos = [
            "Excelente atención, personal amable.",
            "Muy buena experiencia, tratamiento efectivo.",
            "Me sentí acompañado y escuchado.",
            "Rápida atención y soluciones claras."
        ]
        neg = [
            "Mala atención, no volvería.",
            "Me ignoraron, pésima experiencia.",
            "Larga espera y respuestas confusas.",
            "No resolvieron mi problema."
        ]
        rows = []
        for i in range(n):
            if i % 2 == 0:
                rows.append({"text": np.random.choice(pos), "label": 1})
            else:
                rows.append({"text": np.random.choice(neg), "label": 0})
        df = pd.DataFrame(rows)
        return df, "text", "label"

    else:
        raise ValueError(f"dataset.source no soportado: {src}")
