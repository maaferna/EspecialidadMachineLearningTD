# src/explain/lime_shap.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import shap
from lime.lime_text import LimeTextExplainer


def _tidy_title(txt: str, max_chars: int = 90) -> str:
    """Acorta el texto para usarlo en el título del gráfico.
    Elimina saltos de línea y recorta si es muy largo.
    """
    txt = txt.replace("\n", " ").strip()
    if len(txt) <= max_chars:
        return txt
    return textwrap.shorten(txt, width=max_chars, placeholder="…")


def explain_with_lime(
    vectorizer,
    clf,
    texts,
    out_dir,
    topk: int = 10,
    class_names=("neg", "pos"),
):
    """
    Genera explicaciones LIME para textos y guarda:
      - HTML interactivo (lime_ex_{i}.html)
      - PNG (lime_ex_{i}.png) con barras (características + pesos)
    Args:
        vectorizer: Vectorizador (ej. CountVectorizer, TfidfVectorizer).
        clf: Clasificador entrenado (con método predict_proba).
        texts: Lista de textos a explicar.
        out_dir: Directorio donde guardar las explicaciones.
        topk: Número de características a mostrar en el PNG.
        class_names: Nombres de las clases (para LIME).
    Returns: None
    Raises:
        ImportError: Si no está instalado lime.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    explainer = LimeTextExplainer(class_names=list(class_names))

    def predict_proba(xs):
        # Asegura matriz [n,2] para LIME
        proba = clf.predict_proba(vectorizer.transform(xs))
        # Si el clasificador devuelve forma (n,) (raro), lo adaptamos
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        return proba

    for i, t in enumerate(texts):
        exp = explainer.explain_instance(
            t,
            classifier_fn=predict_proba,
            num_features=topk
        )

        # Etiquetas disponibles que LIME pudo explicar (p.ej. [1] en binario)
        avail = exp.available_labels()
        # Predicción del modelo en índices de clases del modelo
        try:
            model_pred_idx = int(clf.predict(vectorizer.transform([t]))[0])
        except Exception:
            model_pred_idx = None

        # Escoge etiqueta segura para LIME
        if model_pred_idx in avail:
            label_idx = model_pred_idx
        else:
            label_idx = avail[0]  # la que LIME tenga disponible (típicamente 1)

        # Guarda HTML
        html_path = out / f"lime_ex_{i}.html"
        exp.save_to_file(str(html_path))

        # También un PNG sencillo (barra con pesos) para el README
        pairs = exp.as_list(label=label_idx)  # [(token, peso), ...]
        feats = [p[0] for p in pairs]
        vals  = [p[1] for p in pairs]
        plt.figure(figsize=(8,4))
        y = np.arange(len(feats))
        plt.barh(y, vals)
        plt.yticks(y, feats)
        plt.title(f"LIME — ej {i} (label {label_idx})")
        plt.tight_layout()
        plt.savefig(out / f"lime_ex_{i}.png", dpi=150)
        plt.close()

def explain_with_shap(vectorizer, clf, texts, out_dir, topk: int = 10):
    """
    Genera explicaciones SHAP “KernelExplainer” y guarda PNGs (bar plot por instancia).
    Args:
        vectorizer: Vectorizador (ej. CountVectorizer, TfidfVectorizer).
        clf: Clasificador entrenado (con método predict_proba).
        texts: Lista de textos a explicar.
        out_dir: Directorio donde guardar las explicaciones.
        topk: Número de características a mostrar en el PNG.
    Returns: None
    Raises:
        ImportError: Si no está instalado shap.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Probabilidad de la clase positiva (índice 1)
    def pred_pos(xs):
        proba = clf.predict_proba(vectorizer.transform(xs))
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        return proba[:, 1]

    # Masker para texto (tokeniza internamente)
    masker = shap.maskers.Text()  # también puedes pasar un tokenizer propio si lo tienes
    explainer = shap.Explainer(pred_pos, masker)

    for i, t in enumerate(texts):
        exp = explainer([t])  # -> shap.Explanation
        vals = np.abs(exp.values[0])     # importancia por token
        toks = exp.data[0]               # tokens alineados

        # top-k tokens por importancia absoluta
        idx = np.argsort(vals)[-topk:]
        plt.figure(figsize=(9, 4.5))
        plt.barh(range(len(idx)), vals[idx])
        plt.yticks(range(len(idx)), [toks[j] for j in idx])
        plt.xlabel("|SHAP value| (importancia)")
        plt.title(f"SHAP — Ej {i}: {_tidy_title(t)}")
        plt.tight_layout()
        plt.savefig(out / f"shap_ex_{i}.png", dpi=160)
        plt.close()