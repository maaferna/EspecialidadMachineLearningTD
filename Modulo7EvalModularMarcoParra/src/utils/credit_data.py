# src/utils/credit_data.py
from __future__ import annotations
import os
import io
import urllib.request
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

COMMON_TARGETS = ["class", "Risk", "Class", "target", "label", "default"]

# Columnas oficiales (20 atributos) + class, con nombres legibles (snake_case)
GERMAN_COLS = [
    "status_checking",          # A1
    "duration_months",          # A2
    "credit_history",           # A3
    "purpose",                  # A4
    "credit_amount",            # A5
    "savings",                  # A6
    "employment_since",         # A7
    "installment_rate",         # A8
    "personal_status_sex",      # A9
    "debtors",                  # A10
    "residence_since",          # A11
    "property",                 # A12
    "age",                      # A13
    "other_installment_plans",  # A14
    "housing",                  # A15
    "existing_credits",         # A16
    "job",                      # A17
    "people_liable",            # A18
    "telephone",                # A19
    "foreign_worker",           # A20
    "class"                     # target (1=good, 2=bad en UCI)
]

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/"
UCI_DATA = UCI_BASE + "german.data"            # 1000 filas, 21 columnas (última es clase)
UCI_NUM  = UCI_BASE + "german.data-numeric"    # opcional: versión numérica
UCI_DOC  = UCI_BASE + "german.doc"             # documentación

def _find_target(df: pd.DataFrame, target: Optional[str]) -> str:
    if target and target in df.columns:
        return target
    for t in COMMON_TARGETS:
        if t in df.columns:
            return t
    raise ValueError(f"No se encontró columna target. Columnas disponibles: {df.columns.tolist()}")

def load_csv(path: str, target: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Carga desde CSV local con target conocido."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe archivo: {path}")
    df = pd.read_csv(path)
    tgt = _find_target(df, target)
    return df, tgt

def load_uci_german(cache_dir: str = ".cache_uci") -> Tuple[pd.DataFrame, str]:
    """
    Descarga 'german.data' desde UCI y construye DataFrame con columnas GERMAN_COLS.
    Mapea la clase 1->'good', 2->'bad' en la columna 'class'.
    """
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, "german.data")

    if not os.path.isfile(local_path):
        print("Descargando UCI german.data ...")
        urllib.request.urlretrieve(UCI_DATA, local_path)

    # Archivo es espacio-separado sin cabecera; 21 columnas (20 features + class)
    df = pd.read_csv(local_path, sep=r"\s+", header=None, names=GERMAN_COLS, engine="python")

    # Target en UCI: 1=good, 2=bad -> convertir a cadenas 'good'/'bad'
    df["class"] = df["class"].map({1: "good", 2: "bad"})
    return df, "class"

def eda_basic(df: pd.DataFrame, target: str) -> Dict:
    y = df[target]
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Distribución de clases (conteo)
    dist = y.value_counts(dropna=False).to_dict()

    info = {
        "n_rows": len(df),
        "n_features": X.shape[1],
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "class_dist": dist,
        "target_name": target
    }
    return info

def prepare_splits(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
):
    # Normalizamos el target a 0/1: impago=1 (bad), no impago=0 (good)
    y_raw = df[target]
    y = y_raw.str.lower().map({"bad": 1, "good": 0}).astype(int)

    X = df.drop(columns=[target])

    # Detectar tipos
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Splits estratificados
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, random_state=random_state, stratify=y_trainval
    )

    # Preprocesamiento: OneHot para categóricas, StandardScaler para numéricas
    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="drop"
    )

    X_train_p = ct.fit_transform(X_train).astype("float32")
    X_val_p   = ct.transform(X_val).astype("float32")
    X_test_p  = ct.transform(X_test).astype("float32")

    # Weights para desbalanceo
    classes = np.array([0, 1])
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.values)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    meta = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "n_features_out": X_train_p.shape[1],
        "class_weight": class_weight_dict
    }
    return (X_train_p, y_train.values), (X_val_p, y_val.values), (X_test_p, y_test.values), ct, meta
