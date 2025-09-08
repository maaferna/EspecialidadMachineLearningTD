"""Pipeline de extremo a extremo: dataset → limpieza → tokens → TF-IDF → métricas y gráficos."""
from __future__ import annotations


import json
import logging
from pathlib import Path
from typing import Dict, List


import pandas as pd


from src.utils.io import ensure_dir, load_yaml, save_json
from src.data.datasets import get_small_clinical_notes
from src.preprocessing.cleaning import basic_clean
from src.preprocessing.stopwords import get_stopwords
from src.preprocessing.tokenize_spacy import SpacyProcessor
from src.preprocessing.tokenize_nltk import nltk_tokens
from src.features.vectorize_tfidf import fit_tfidf, top_terms_by_doc, vocab_to_file
from src.evaluation.corpus_stats import compare_corpora
from src.visualization.term_bars import plot_top_terms_for_doc


logger = logging.getLogger(__name__)




def _prep_texts(df: pd.DataFrame, text_col: str, clean_cfg: dict) -> List[str]:
    return [
    basic_clean(
    str(x),
    lowercase=clean_cfg.get("lowercase", True),
    remove_urls=clean_cfg.get("remove_urls", True),
    remove_emails=clean_cfg.get("remove_emails", True),
    remove_numbers=clean_cfg.get("remove_numbers", True),
    remove_punct=clean_cfg.get("remove_punct", True),
    )
    for x in df[text_col].tolist()
    ]




def run(cfg_path: str | Path = "configs/config_default.yaml") -> Dict:
    cfg = load_yaml(cfg_path)


    # 1) Dataset
    df = get_small_clinical_notes(cfg)
    text_col = cfg["dataset"]["text_column"]


    # 2) Limpieza básica
    cleaned_texts = _prep_texts(df, text_col, cfg.get("preprocessing", {}))


    # 3) Stopwords
    sw = get_stopwords(
    language=cfg["preprocessing"].get("language", "es"),
    source=cfg["preprocessing"].get("stopwords_source", "spacy"),
    )


    # 4) Tokens spaCy
    sp = SpacyProcessor(
    language=cfg["preprocessing"].get("language", "es"),
    model_es=cfg["spacy"].get("model_es", "es_core_news_sm"),
    model_en=cfg["spacy"].get("model_en", "en_core_web_sm"),
    keep_pos=cfg["preprocessing"].get("keep_pos", ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]),
    stopwords=sw,
    )
    spacy_tokens = [sp.tokens(txt) for txt in cleaned_texts]
    spacy_docs = [" ".join(toks) for toks in spacy_tokens]


    # 5) Tokens NLTK
    nltk_tokens_list = [
    nltk_tokens(
    txt,
    language=cfg["preprocessing"].get("language", "es"),
    normalize=cfg["nltk"].get("normalize", "stem"),
    stopwords=sw,
    )
    for txt in cleaned_texts
    ]
    nltk_docs = [" ".join(toks) for toks in nltk_tokens_list]

    # 6) TF-IDF para cada enfoque
    vec_cfg = cfg["vectorizer"]
    vec_spacy, X_spacy = fit_tfidf(
    spacy_docs,
    max_features=vec_cfg.get("max_features", 2000),
    ngram_range=tuple(vec_cfg.get("ngram_range", [1, 2])),
    min_df=vec_cfg.get("min_df", 1),
    max_df=vec_cfg.get("max_df", 0.95),
    )
    vec_nltk, X_nltk = fit_tfidf(
    nltk_docs,
    max_features=vec_cfg.get("max_features", 2000),
    ngram_range=tuple(vec_cfg.get("ngram_range", [1, 2])),
    min_df=vec_cfg.get("min_df", 1),
    max_df=vec_cfg.get("max_df", 0.95),
    )
    # 7) Métricas corpus
    metrics = {
    "spacy": compare_corpora([t.split() for t in cleaned_texts], spacy_tokens),
    "nltk": compare_corpora([t.split() for t in cleaned_texts], nltk_tokens_list),
    }


    # 8) Guardado de artefactos (vocabularios y gráficos)
    out_cfg = cfg["outputs"]
    ensure_dir(Path(cfg["project"]["figures_dir"]))
    vocab_to_file(vec_spacy, out_cfg.get("vocab_spacy_txt", "data/processed/vocab_spacy.txt"))
    vocab_to_file(vec_nltk, out_cfg.get("vocab_nltk_txt", "data/processed/vocab_nltk.txt"))


    top_k = cfg["visualization"].get("top_k_terms", 10)
    spacy_top = top_terms_by_doc(X_spacy, vec_spacy, top_k=top_k)
    nltk_top = top_terms_by_doc(X_nltk, vec_nltk, top_k=top_k)


    for i, terms_scores in enumerate(spacy_top):
        plot_top_terms_for_doc(
        terms_scores,
        out_path=Path(cfg["project"]["figures_dir"]) / f"doc_{i+1:02d}_spacy.png",
        title=f"Doc {i+1} — spaCy",
        )


    for i, terms_scores in enumerate(nltk_top):
        plot_top_terms_for_doc(
        terms_scores,
        out_path=Path(cfg["project"]["figures_dir"]) / f"doc_{i+1:02d}_nltk.png",
        title=f"Doc {i+1} — NLTK",
        )


    # 9) Persistir métricas
    save_json(metrics, cfg["outputs"].get("metrics_json", "reports/metrics.json"))
    logger.info("Métricas guardadas en %s", cfg["outputs"].get("metrics_json"))


    return metrics