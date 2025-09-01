# scripts/main.py
# -*- coding: utf-8 -*-
"""
Sesión 1 — NLP clásico (BoW / TF-IDF / n-gramas)
- Carga corpus
- Vectoriza (BoW y TF-IDF) con opciones de n-gramas y DF
- Similaridad por coseno (TF-IDF)
- Heatmap + Top términos por documento
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── módulos del proyecto ──────────────────────────────────────────────────────
from src.data.loader import load_corpus
from src.features.vectorize import vectorize_all, top_terms_for_doc
from src.visualizer.plots import (
    plot_similarity_heatmap,
    plot_top_terms_bars,
)


# ── utilidades internas ──────────────────────────────────────────────────────
def ensure_dir(path: str | os.PathLike) -> None:
    """Crea la carpeta si no existe (idempotente)."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detectando similitud y términos clave en textos clínicos"
    )

    # I/O
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/clinical_notes.txt",
        help="Ruta al archivo de texto (una nota por línea).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Carpeta de salida para figuras.",
    )

    # Vectorización
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="n-gram mínimo (1 = unigramas).",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=1,
        help="n-gram máximo (1 = unigramas).",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Frecuencia documental mínima (entero).",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.85,
        help="Frecuencia documental máxima (proporción).",
    )

    # Reportes
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Cantidad de términos top por documento (TF-IDF).",
    )
    parser.add_argument(
        "--docs-top",
        type=int,
        default=3,
        help="Número de documentos a los que graficar términos top.",
    )

    return parser.parse_args()


# ── Lógica principal ─────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    # 1) Cargar corpus
    docs: List[str] = load_corpus(args.corpus)
    print(f"Corpus cargado: {len(docs)} documentos")

    # 2) Vectorización (BoW y TF-IDF) con tus parámetros
    X_bow, X_tfidf, feat_names, _, _ = vectorize_all(
        docs=docs,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
        norm="l2",
    )

    print(
        f"BoW shape: {X_bow.shape} | TF-IDF shape: {X_tfidf.shape}"
    )

    # 3) Similaridad por coseno (TF-IDF) y heatmap
    sim: np.ndarray = cosine_similarity(X_tfidf)
    heatmap_path = os.path.join(args.out_dir, "similarity_heatmap.png")
    plot_similarity_heatmap(sim, out_path=heatmap_path)
    print(f"Heatmap: {heatmap_path}")

    # 4) Top términos por documento (TF-IDF)
    n_docs = min(args.docs_top, X_tfidf.shape[0])
    for d in range(n_docs):
        top = top_terms_for_doc(X_tfidf, feat_names, doc_idx=d, k=args.topk)
        out_png = os.path.join(args.out_dir, f"doc{d}_top_terms.png")
        plot_top_terms_bars(top, title=f"Doc {d} - Top {args.topk} términos (TF-IDF)",path=out_png)
        print(f"Top términos Doc {d}: {out_png}")

    # 5) Mini ranking de pares más similares (opcional, visual rápido)
    #    Mostramos el mejor vecino para cada documento (excluyendo la diagonal)
    print("\nDocumentos más similares (TF-IDF + coseno):")
    for i in range(sim.shape[0]):
        # poner a -inf la diagonal para ignorar el self-match
        row = sim[i].copy()
        row[i] = -np.inf
        j = int(np.argmax(row))
        print(f"  D{i} ↔ D{j} | similitud={sim[i, j]:.3f}")


if __name__ == "__main__":
    main()
