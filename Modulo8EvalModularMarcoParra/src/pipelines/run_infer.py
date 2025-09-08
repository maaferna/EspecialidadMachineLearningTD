"""Pipeline de inferencia: carga artefactos y predice."""
from __future__ import annotations
from pathlib import Path
from typing import List
from src.utils.config import load_yaml
from src.preprocessing.cleaning import basic_clean
from src.preprocessing.stopwords import get_stopwords
from src.preprocessing.tokenize_spacy import SpacyTokenizer
from src.preprocessing.tokenize_nltk import nltk_tokenize
from src.models.predict import load_artifacts, predict_texts




def build_preprocess_fn(cfg: dict):
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
    tokenizer = cfg["preprocessing"].get("tokenizer", "spacy").lower()
    keep_pos = cfg["preprocessing"].get("keep_pos", ["NOUN","VERB","ADJ","ADV","PROPN"])


    if tokenizer == "nltk":
        def preprocess(text: str) -> str:
            t = basic_clean(text)
            return " ".join(nltk_tokenize(t, language=lang, normalize=cfg["preprocessing"].get("nltk_normalize","stem"), stopwords=sw))
        return preprocess
    else:
        sp = SpacyTokenizer(language=lang, keep_pos=keep_pos, stopwords=sw)
        def preprocess(text: str) -> str:
            return " ".join(sp(basic_clean(text)))
        return preprocess




def run(texts: List[str], cfg_path: str | Path = "configs/config_default.yaml"):
    """Ejecuta pipeline de inferencia: carga artefactos y predice etiquetas para `texts`.
    Parameters
    ----------
    texts : List[str]
        Lista de textos a clasificar.
    cfg_path : str | Path
        Ruta al archivo de configuración YAML.
    Returns"""
    cfg = load_yaml(cfg_path)
    vec, clf, le = load_artifacts(cfg["outputs"]["vectorizer_path"], cfg["outputs"]["model_path"], cfg["outputs"]["label_encoder_path"])
    preprocess = build_preprocess_fn(cfg)
    labels, ypred = predict_texts(texts, preprocess, vec, clf, le)
    return labels