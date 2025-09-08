"""Pipeline de entrenamiento: carga datos → preprocesa → TF-IDF → entrena → evalúa → guarda artefactos."""

from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


from src.utils.config import load_yaml
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.data.datasets import load_dataset
from src.data.splits import make_splits
from src.preprocessing.cleaning import basic_clean
from src.preprocessing.stopwords import get_stopwords
from src.preprocessing.tokenize_spacy import SpacyTokenizer
from src.preprocessing.tokenize_nltk import nltk_tokenize
from src.features.vectorize_tfidf import fit_tfidf
from src.models.train_classifier import train_tfidf_classifier, save_artifacts
from src.evaluation.metrics import summary_metrics
from src.visualization.plots import plot_confusion


def _tokens_list(texts, cfg):
    """Tokeniza lista de textos según configuración `cfg`.
    Soporta tokenización con spacy o nltk + stopwords + POS tagging (spacy"""
    lang = cfg["preprocessing"].get("language","es")
    sw = get_stopwords(lang, cfg["preprocessing"].get("stopwords_source","spacy"))
    tok = cfg["preprocessing"].get("tokenizer","spacy")
    keep_pos = cfg["preprocessing"].get("keep_pos", ["NOUN","VERB","ADJ","ADV","PROPN"])
    if tok == "nltk":
        return [nltk_tokenize(basic_clean(t), language=lang, normalize=cfg["preprocessing"].get("nltk_normalize","stem"), stopwords=sw) for t in texts]
    else:
        sp = SpacyTokenizer(language=lang, keep_pos=keep_pos, stopwords=sw)
        return [sp(basic_clean(t)) for t in texts]

def build_preprocess_fn(cfg: dict) -> Callable[[str], str]:
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
            t = basic_clean(text,
                lowercase=cfg["preprocessing"].get("lowercase", True),
                remove_urls=cfg["preprocessing"].get("remove_urls", True),
                remove_emails=cfg["preprocessing"].get("remove_emails", True),
                remove_numbers=cfg["preprocessing"].get("remove_numbers", True),
                remove_punct=cfg["preprocessing"].get("remove_punct", True))
            toks = nltk_tokenize(t, language=lang, normalize=cfg["preprocessing"].get("nltk_normalize","stem"), stopwords=sw)
            return " ".join(toks)
        return preprocess
    else:
        sp = SpacyTokenizer(language=lang, keep_pos=keep_pos, stopwords=sw)
        def preprocess(text: str) -> str:
            t = basic_clean(text,
                lowercase=cfg["preprocessing"].get("lowercase", True),
                remove_urls=cfg["preprocessing"].get("remove_urls", True),
                remove_emails=cfg["preprocessing"].get("remove_emails", True),
                remove_numbers=cfg["preprocessing"].get("remove_numbers", True),
                remove_punct=cfg["preprocessing"].get("remove_punct", True))
            return " ".join(sp(t))
        return preprocess



from pathlib import Path
from typing import Union
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import load_yaml
from src.utils.io import save_json
from src.utils.logging import setup_logging


from src.data.datasets import load_dataset
from src.data.splits import make_splits


from src.visualization.plots import plot_confusion


# NUEVO: selectores de features/modelos
from src.features.embeddings import train_word2vec, train_fasttext, docs_to_matrix
from src.features.transformer_embedder import load_st_model, encode_texts
from src.models.classifiers import build_classifier


from pathlib import Path
from typing import Union, Optional, List
from datetime import datetime
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import load_yaml
from src.utils.io import save_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.data.datasets import load_dataset
from src.data.splits import make_splits

from src.visualization.plots import plot_confusion

# NUEVO: selectores de features/modelos
from src.features.embeddings import train_word2vec, train_fasttext, docs_to_matrix
from src.features.transformer_embedder import load_st_model, encode_texts
from src.models.classifiers import build_classifier


def _texts_to_tokens(texts: List[str], preprocess_fn) -> List[List[str]]:
    """Aplica tu función modular de preprocesamiento y devuelve lista de tokens."""
    # Asumimos que preprocess_fn(texto) retorna un string 'token1 token2 ...'
    return [preprocess_fn(t).split() for t in texts]


def run(
    cfg_path: Union[str, Path] = "configs/config_default.yaml",
    override_model: Optional[str] = None,
    override_rep: Optional[str] = None,
    run_tag: str = "",
) -> dict:
    """
    Ejecuta pipeline completo de entrenamiento y evaluación con soporte para:
      - Representación: TF-IDF | Word2Vec | FastText | Transformer embeddings (Sentence-Transformers)
      - Modelos: multinomial_nb | logreg | linear_svm
    Además:
      - Permite overrides por CLI (modelo/representación)
      - Crea un run_dir único por ejecución para no sobrescribir artefactos
    """
    # --- Config y reproducibilidad
    cfg = load_yaml(cfg_path)
    set_seed(cfg["project"].get("seed", 42))
    setup_logging(cfg["project"]["logs_dir"], level="INFO")

    # --- Overrides desde CLI (opcional)
    if override_model:
        cfg.setdefault("model", {})["name"] = override_model
    if override_rep:
        cfg.setdefault("features", {})["vectorizer"] = override_rep

    # --- Datos
    df, text_col, label_col, sens_col = load_dataset(cfg)
    train_df, test_df = make_splits(
        df,
        label_col,
        test_size=cfg["dataset"].get("test_size", 0.2),
        stratify=cfg["dataset"].get("stratify", True),
        random_state=cfg["project"].get("seed", 42),
    )

    # --- Preprocesador (pipeline modular existente)
    preprocess = build_preprocess_fn(cfg)

    # --- y: encoding robusto (acepta string o numérico)
    y_train_raw = train_df[label_col].to_numpy()
    y_test_raw = test_df[label_col].to_numpy()
    le = LabelEncoder()
    try:
        y_train = le.fit_transform(y_train_raw)
        y_test = le.transform(y_test_raw)
    except Exception:
        # si ya son numéricos homogéneos
        le.classes_ = np.unique(y_train_raw)
        y_train = y_train_raw.astype(int)
        y_test = y_test_raw.astype(int)

    # --- X: selección de representación
    rep = cfg["features"].get("vectorizer", "tfidf").lower()
    vec_artifact = None  # lo que persistiremos como "vectorizer" (o modelo de embeddings)

    train_texts = train_df[text_col].astype(str).tolist()
    test_texts = test_df[text_col].astype(str).tolist()

    if rep == "tfidf":
        # Preprocesa a "texto limpio" (tokens unidos por espacio)
        X_train_clean = [" ".join(preprocess(t).split()) for t in train_texts]
        X_test_clean = [" ".join(preprocess(t).split()) for t in test_texts]

        vec = TfidfVectorizer(
            max_features=cfg["features"]["tfidf"].get("max_features", 5000),
            ngram_range=tuple(cfg["features"]["tfidf"].get("ngram_range", [1, 2])),
            min_df=cfg["features"]["tfidf"].get("min_df", 1),
            max_df=cfg["features"]["tfidf"].get("max_df", 0.95),
        )
        X_train = vec.fit_transform(X_train_clean)
        X_test = vec.transform(X_test_clean)
        vec_artifact = vec

    elif rep in ("word2vec", "fasttext"):
        # Tokens (manteniendo tu limpieza/lemma/stopwords)
        train_tokens = _texts_to_tokens(train_texts, preprocess)
        test_tokens = _texts_to_tokens(test_texts, preprocess)

        if rep == "word2vec":
            w2v = train_word2vec(
                train_tokens,
                size=cfg["features"]["word2vec"].get("size", 100),
                window=cfg["features"]["word2vec"].get("window", 5),
                min_count=cfg["features"]["word2vec"].get("min_count", 2),
                sg=cfg["features"]["word2vec"].get("sg", 1),
            )
            X_train = docs_to_matrix(train_tokens, w2v, normalize=True)
            X_test = docs_to_matrix(test_tokens, w2v, normalize=True)
            vec_artifact = w2v

        else:  # fasttext
            ft = train_fasttext(
                train_tokens,
                size=cfg["features"]["fasttext"].get("size", 100),
                window=cfg["features"]["fasttext"].get("window", 5),
                min_count=cfg["features"]["fasttext"].get("min_count", 2),
                sg=cfg["features"]["fasttext"].get("sg", 1),
            )
            X_train = docs_to_matrix(train_tokens, ft, normalize=True)
            X_test = docs_to_matrix(test_tokens, ft, normalize=True)
            vec_artifact = ft

    elif rep == "transformer_embed":
        # Embeddings BERT (rápidos, sin fine-tuning) con Sentence-Transformers
        st_name = cfg["features"]["transformer_embed"].get(
            "model", "sentence-transformers/distiluse-base-multilingual-cased-v2"
        )
        batch_size = cfg["features"]["transformer_embed"].get("batch_size", 32)
        st_model = load_st_model(st_name)
        X_train = encode_texts(st_model, train_texts, batch_size=batch_size)
        X_test = encode_texts(st_model, test_texts, batch_size=batch_size)
        vec_artifact = {"st_model": st_name}  # guardamos el nombre como metadato

    else:
        raise ValueError(f"Representación no soportada: {rep}")

    # --- Modelo (incluye multinomial_nb | logreg | linear_svm)
    model_name = cfg["model"]["name"].lower()
    clf = build_classifier(model_name, cfg)

    # Validación rápida: MNB requiere entradas no negativas (TF-IDF/BoW)
    if model_name == "multinomial_nb" and rep != "tfidf":
        raise ValueError(
            "MultinomialNB solo es compatible con TF-IDF/BoW (valores no negativos). "
            "Usa logreg/linear_svm con embeddings (w2v/ft/transformer)."
        )

    # --- Entrenamiento
    clf.fit(X_train, y_train)

    # --- Evaluación
    y_pred = clf.predict(X_test)
    labels_order = list(le.classes_)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=[str(x) for x in labels_order],
        output_dict=True,
        zero_division=0,
    )
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # --- Crear run_dir único (no sobrescribir)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tag = f"_{run_tag}" if run_tag else ""
    base_out = Path(cfg["project"].get("output_dir", "reports"))
    run_dir = base_out / "runs" / f"{ts}_{model_name}_{rep}{tag}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "fairness").mkdir(parents=True, exist_ok=True)

    # --- Reubicar outputs a la carpeta del run
    cfg["outputs"]["metrics_json"] = str(run_dir / "metrics_cls.json")
    cfg["outputs"]["confusion_png"] = str(run_dir / "figures" / "confusion_matrix.png")
    cfg["outputs"]["model_path"] = str(run_dir / "models" / "model.joblib")
    cfg["outputs"]["vectorizer_path"] = str(run_dir / "models" / "vectorizer.joblib")
    cfg["outputs"]["label_encoder_path"] = str(run_dir / "models" / "label_encoder.joblib")

    # --- Guardado de métricas y figuras
    metrics = {
        "representation": rep,
        "model": model_name,
        "accuracy": acc,
        "f1_macro": f1m,
        "report": report,
        "confusion": cm.tolist(),
        "support": int(cm.sum()),
        "run_dir": str(run_dir),
    }
    save_json(metrics, cfg["outputs"]["metrics_json"])

    # plot_confusion acepta (cm, labels, out_path, title=None). Si tu versión no admite `title`,
    # quita el argumento `title=` de la siguiente llamada.
    plot_confusion(
        np.array(metrics["confusion"]),
        [str(x) for x in labels_order],
        cfg["outputs"]["confusion_png"],
        title=f"Confusion — {model_name} | {rep}",
    )

    # --- Artefactos (vectorizer/embeddings, modelo y label encoder)
    save_artifacts(
        vec_artifact,
        clf,
        le,
        cfg["outputs"]["vectorizer_path"],
        cfg["outputs"]["model_path"],
        cfg["outputs"]["label_encoder_path"],
    )

    print(f"[INFO] Run guardada en: {run_dir}")
    return metrics
