"""
Explicabilidad de predicciones clínicas con SHAP y LIME.
Genera visualizaciones para un subconjunto de ejemplos de test.
"""

from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer

from src.utils.config import load_yaml
from src.preprocessing.cleaning import basic_clean
from src.preprocessing.stopwords import get_stopwords
from src.preprocessing.tokenize_spacy import SpacyTokenizer
from src.preprocessing.tokenize_nltk import nltk_tokenize
from src.data.datasets import load_dataset
from src.data.splits import make_splits

def _build_preprocess_fn(cfg: dict):
    lang = cfg["preprocessing"].get("language", "es")
    sw = get_stopwords(lang, cfg["preprocessing"].get("stopwords_source", "spacy"))
    tok = cfg["preprocessing"].get("tokenizer", "spacy").lower()
    keep_pos = cfg["preprocessing"].get(
        "keep_pos", ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]
    )

    if tok == "nltk":
        return lambda s: " ".join(
            nltk_tokenize(
                basic_clean(s),
                language=lang,
                normalize=cfg["preprocessing"].get("nltk_normalize", "stem"),
                stopwords=sw,
            )
        )
    else:
        sp = SpacyTokenizer(language=lang, keep_pos=keep_pos, stopwords=sw)
        return lambda s: " ".join(sp(basic_clean(s)))


def explain_with_shap(cfg: dict, texts: list[str], out_dir: Path):
    """
    SHAP para modelos lineales (TF-IDF). Maneja binario/multiclase.
    - Usa background de TRAIN (mejor que 'dummy')
    - Extrae arrays desde shap.Explanation
    - En multiclase, toma la clase predicha por muestra
    - Dibuja barplots top-k por documento
    """
    import numpy as np
    import shap, joblib, matplotlib.pyplot as plt
    from src.data.datasets import load_dataset
    from src.data.splits import make_splits

    # Artefactos
    vec = joblib.load(cfg["outputs"]["vectorizer_path"])
    clf = joblib.load(cfg["outputs"]["model_path"])
    feat_names = np.array(getattr(vec, "get_feature_names_out")())

    # Prepro
    preprocess = _build_preprocess_fn(cfg)

    # Background: train
    df, text_col, label_col, _ = load_dataset(cfg)
    train_df, test_df = make_splits(
        df,
        label_col,
        test_size=cfg["dataset"].get("test_size", 0.3),
        stratify=cfg["dataset"].get("stratify", True),
        random_state=cfg["project"].get("seed", 42),
    )
    bg_texts = train_df[text_col].astype(str).tolist()[:200]
    X_bg = vec.transform([preprocess(t) for t in bg_texts])

    # Matriz a explicar
    X = vec.transform([preprocess(t) for t in texts])

    out_dir.mkdir(parents=True, exist_ok=True)

    # Helper para multiclase si no hay predict_proba
    def _softmax(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores)
        exps = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exps / exps.sum(axis=1, keepdims=True)

    # Predicción de clase por muestra
    if hasattr(clf, "predict_proba"):
        yhat = np.argmax(clf.predict_proba(X), axis=1)
    elif hasattr(clf, "decision_function"):
        yhat = np.argmax(clf.decision_function(X), axis=1)
    else:
        yhat = clf.predict(X)  # último recurso

    # Explainer lineal (rápido). Si falla, caemos a KernelExplainer (omitido por brevedad).
    explainer = shap.LinearExplainer(clf, X_bg)
    sv = explainer(X)  # sv es shap.Explanation (reciente) o lista (legacy)

    TOPK = 15
    n = X.shape[0]

    # Caso 1: shap.Explanation (reciente)
    if isinstance(sv, shap._explanation.Explanation):
        values = sv.values  # shape: (n_samples, n_features) o (n_samples, n_features, n_classes)
        # Binario: (n, f)
        if values.ndim == 2:
            for i in range(n):
                row_vals = values[i]  # (f,)
                idx = np.argsort(np.abs(row_vals))[-TOPK:]
                terms = feat_names[idx]; vals = row_vals[idx]
                order = np.argsort(np.abs(vals))
                plt.figure(figsize=(7, 4.5))
                plt.barh(range(len(idx)), vals[order])
                plt.yticks(range(len(idx)), terms[order])
                plt.title(f"SHAP — Ej {i}: {texts[i][:60]}...")
                plt.tight_layout()
                out_file = out_dir / f"shap_ex_{i}.png"
                plt.savefig(out_file, dpi=160, bbox_inches="tight")
                plt.close()
                print(f"[SHAP] Guardado: {out_file}")
        # Multiclase: (n, f, C)
        elif values.ndim == 3:
            for i in range(n):
                c = int(yhat[i])
                row_vals = values[i, :, c]  # (f,)
                idx = np.argsort(np.abs(row_vals))[-TOPK:]
                terms = feat_names[idx]; vals = row_vals[idx]
                order = np.argsort(np.abs(vals))
                plt.figure(figsize=(7, 4.5))
                plt.barh(range(len(idx)), vals[order])
                plt.yticks(range(len(idx)), terms[order])
                plt.title(f"SHAP — Ej {i} (clase {c}): {texts[i][:60]}...")
                plt.tight_layout()
                out_file = out_dir / f"shap_ex_{i}.png"
                plt.savefig(out_file, dpi=160, bbox_inches="tight")
                plt.close()
                print(f"[SHAP] Guardado: {out_file}")
        else:
            raise RuntimeError(f"Dimensiones SHAP no soportadas: {values.shape}")

    else:
        # Caso 2: legacy (lista por clase) -> sv[c] = (n, f)
        # Elegimos la clase predicha por muestra
        for i in range(n):
            c = int(yhat[i])
            row_vals = sv[c][i]  # (f,)
            idx = np.argsort(np.abs(row_vals))[-TOPK:]
            terms = feat_names[idx]; vals = row_vals[idx]
            order = np.argsort(np.abs(vals))
            plt.figure(figsize=(7, 4.5))
            plt.barh(range(len(idx)), vals[order])
            plt.yticks(range(len(idx)), terms[order])
            plt.title(f"SHAP — Ej {i} (clase {c}): {texts[i][:60]}...")
            plt.tight_layout()
            out_file = out_dir / f"shap_ex_{i}.png"
            plt.savefig(out_file, dpi=160, bbox_inches="tight")
            plt.close()
            print(f"[SHAP] Guardado: {out_file}")

# --- dentro de src/pipelines/run_explainability.py ---
from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

def explain_with_lime(cfg: dict, texts: list[str], out_dir: Path, save_png: bool = True):
    """
    Genera explicaciones LIME en HTML (interactivo) y PNG (estático) por cada ejemplo.
    - HTML: lime_ex_{i}.html
    - PNG : lime_ex_{i}.png (si save_png=True)
    """
    # 1) Cargar artefactos
    vec_path = cfg["outputs"]["vectorizer_path"]
    clf_path = cfg["outputs"]["model_path"]
    vec = joblib.load(vec_path)
    clf = joblib.load(clf_path)

    # 2) Preprocesador consistente con el train
    try:
        preprocess = _build_preprocess_fn(cfg)  # ya existe en tu pipeline
    except NameError:
        preprocess = lambda s: s

    # 3) Función predict_proba sobre texto crudo
    def predict_proba(raw_texts):
        X = vec.transform([preprocess(t) for t in raw_texts])
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X)
            exps = np.exp(scores - scores.max(axis=1, keepdims=True))
            return exps / exps.sum(axis=1, keepdims=True)
        # Fallback simple
        y = clf.predict(X).astype(int)
        n_classes = len(np.unique(y))
        out = np.zeros((len(y), n_classes))
        out[np.arange(len(y)), y] = 1.0
        return out

    class_names = cfg.get("labels", ["0", "1", "2"])
    explainer = LimeTextExplainer(class_names=class_names)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) Explicar cada texto
    for i, text in enumerate(texts):
        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba,
            num_features=15,
            labels=list(range(len(class_names)))
        )

        # 4a) Guardar HTML interactivo
        html_path = out_dir / f"lime_ex_{i}.html"
        exp.save_to_file(str(html_path))
        print(f"[LIME] Guardado: {html_path}")

        if not save_png:
            continue

        # 4b) PNG para README: usamos la clase predicha y barplot de contribuciones
        yhat = int(np.argmax(predict_proba([text])[0]))
        weights = exp.as_list(label=yhat) or []
        if not weights:
            # Si LIME no devuelve pares (token, peso), saltamos
            continue

        tokens, vals = zip(*weights)
        tokens = np.array(tokens)
        vals = np.array(vals, dtype=float)

        # Orden por contribución absoluta (pequeños abajo, grandes arriba)
        order = np.argsort(np.abs(vals))
        tokens = tokens[order]
        vals = vals[order]

        plt.figure(figsize=(10, 5))
        plt.barh(range(len(vals)), vals)
        plt.yticks(range(len(vals)), tokens)
        # recorta título para no desbordar
        short_text = (text[:90] + "…") if len(text) > 90 else text
        plt.title(f"LIME — Ej {i} (clase {yhat}): {short_text}")
        plt.tight_layout()
        png_path = out_dir / f"lime_ex_{i}.png"
        plt.savefig(png_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[LIME] PNG Guardado: {png_path}")



def run(cfg_path: str = "configs/config_default.yaml", method: str = "shap", n_samples: int = 3):
    cfg = load_yaml(cfg_path)
    df, text_col, label_col, sens_col = load_dataset(cfg)
    train_df, test_df = make_splits(
        df,
        label_col,
        test_size=cfg["dataset"].get("test_size", 0.3),
        stratify=cfg["dataset"].get("stratify", True),
        random_state=cfg["project"].get("seed", 42),
    )

    texts = test_df[text_col].astype(str).tolist()[:n_samples]
    out_dir = Path(cfg["project"]["output_dir"]) / "figures" / "explainability"

    if method == "shap":
        explain_with_shap(cfg, texts, out_dir)
    else:
        explain_with_lime(cfg, texts, out_dir)
