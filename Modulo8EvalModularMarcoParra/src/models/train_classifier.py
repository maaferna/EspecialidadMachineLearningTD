"""Entrenamiento de clasificador lineal con TF-IDF y opciones de re-muestreo."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


from src.features.vectorize_tfidf import fit_tfidf
from src.utils.io import ensure_dir


def build_xy(df: pd.DataFrame, text_col: str, label_col: str, preprocess_fn) -> Tuple[list[str], list[str]]:
    """Extrae listas de textos y etiquetas desde el DataFrame."""
    texts = df[text_col].astype(str).tolist()
    texts = [preprocess_fn(t) for t in texts]
    labels = df[label_col].astype(str).tolist()
    return texts, labels


def train_tfidf_classifier(
    train_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    preprocess_fn,
    cfg: dict,
    ):
    """Entrena un clasificador lineal (LogReg o LinearSVC) con TF-IDF y opciones de re-muestreo.
    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame de entrenamiento con columnas de texto y etiqueta.
    text_col : str
        Nombre de la columna de texto.
    label_col : str
        Nombre de la columna de etiquetas.
    preprocess_fn : Callable[[str], str]
        Función de preprocesamiento de texto.
    cfg : dict
        Configuración del modelo y características.
        Debe incluir:
        - features: configuración de TF-IDF (max_features, ngram_range, min_df, max_df)
        - model: configuración del modelo (name, class_weight, C, resampling)
    Returns
    -------
    Tuple[Pipeline, ClassifierMixin, LabelEncoder]
        Pipeline de vectorización, clasificador entrenado y codificador de etiquetas.
    """

    # Prepara textos y etiquetas
    X_texts, y_str = build_xy(train_df, text_col, label_col, preprocess_fn)


    # LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_str)


    # TF-IDF
    tf_cfg = cfg["features"]["tfidf"] # Configuración de TF-IDF
    vec, X = fit_tfidf(
        X_texts,
        max_features=tf_cfg.get("max_features", 5000),
        ngram_range=tuple(tf_cfg.get("ngram_range", [1,2])),
        min_df=tf_cfg.get("min_df", 1),
        max_df=tf_cfg.get("max_df", 0.95),
        ) # X es csr_matrix


    # Remuestreo opcional
    resampling = cfg["model"].get("resampling", "none") # Permite gestionar clases desbalanceadas
    if resampling == "undersample":
        X, y = RandomUnderSampler(random_state=42).fit_resample(X, y)
    elif resampling == "oversample":
        X, y = RandomOverSampler(random_state=42).fit_resample(X, y)


    # Modelo
    name = cfg["model"].get("name", "linear_svm")
    class_weight = None if cfg["model"].get("class_weight", "none") == "none" else "balanced"
    C = float(cfg["model"].get("C", 1.0))


    if name == "logreg":
        clf = LogisticRegression(max_iter=200, n_jobs=None, class_weight=class_weight, C=C)
    else:
        clf = LinearSVC(class_weight=class_weight, C=C)


    clf.fit(X, y)


    return vec, clf, le


def save_artifacts(vec, clf, le, vec_path: str | Path, model_path: str | Path, le_path: str | Path) -> None:
    """Guarda artefactos del modelo en disco (vectorizador, clasificador, label encoder).
    Utiliza joblib."""
    ensure_dir(Path(vec_path).parent)
    joblib.dump(vec, vec_path)
    joblib.dump(clf, model_path)
    joblib.dump(le, le_path)