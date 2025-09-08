"""Auditoría simple de fairness: métricas por grupo sensible en test."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.config import load_yaml
from src.utils.io import save_json
from src.data.datasets import load_dataset
from src.data.splits import make_splits
from src.preprocessing.cleaning import basic_clean
from src.preprocessing.stopwords import get_stopwords
from src.preprocessing.tokenize_spacy import SpacyTokenizer
from src.preprocessing.tokenize_nltk import nltk_tokenize
from src.models.predict import load_artifacts
from src.evaluation.fairness import group_metrics




def _build_preprocess_fn(cfg: dict):
    """Construye función de preprocesamiento según configuración `cfg`.
    Soporta limpieza básica + tokenización (spacy o nltk) + stopwords + POS tagging (spacy).
    Parameters
    ----------
    cfg : dict
        Configuración (sección "preprocessing" del YAML).
    Returns
    -------
    Callable[[str], str]
        Función que recibe un string y devuelve un string preprocesado.
    -------
    """
    lang = cfg["preprocessing"].get("language", "es")
    sw = get_stopwords(lang, cfg["preprocessing"].get("stopwords_source", "spacy"))
    tok = cfg["preprocessing"].get("tokenizer", "spacy").lower()
    keep_pos = cfg["preprocessing"].get("keep_pos", ["NOUN","VERB","ADJ","ADV","PROPN"])
    if tok == "nltk":
        return lambda s: " ".join(nltk_tokenize(basic_clean(s), language=lang, normalize=cfg["preprocessing"].get("nltk_normalize","stem"), stopwords=sw))
    sp = SpacyTokenizer(language=lang, keep_pos=keep_pos, stopwords=sw)
    return lambda s: " ".join(sp(basic_clean(s)))




def run(cfg_path: str | Path = "configs/config_default.yaml") -> dict:
    cfg = load_yaml(cfg_path)
    df, text_col, label_col, sens_col = load_dataset(cfg)
    if not sens_col:
        raise ValueError("No se definió una columna sensible en el dataset para la auditoría.")
    train_df, test_df = make_splits(df, label_col, test_size=cfg["dataset"].get("test_size",0.2), stratify=cfg["dataset"].get("stratify", True), random_state=cfg["project"].get("seed",42))


    preprocess = _build_preprocess_fn(cfg)
    vec, clf, le = load_artifacts(cfg["outputs"]["vectorizer_path"], cfg["outputs"]["model_path"], cfg["outputs"]["label_encoder_path"])


    X_proc = [preprocess(t) for t in test_df[text_col].astype(str).tolist()]
    y_true = test_df[label_col].astype(str).tolist()


    # Predicción
    Xv = vec.transform(X_proc)
    y_pred = clf.predict(Xv)
    y_pred_lbl = le.inverse_transform(y_pred)


    out_df = test_df.copy()
    out_df["y_pred"] = y_pred_lbl


    metrics = group_metrics(out_df, y_true_col=label_col, y_pred_col="y_pred", group_col=sens_col)
    save_json(metrics, cfg["outputs"]["fairness_json"])
    return metrics