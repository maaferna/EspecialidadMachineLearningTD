"""Vectorización TF-IDF y utilidades para extraer términos por documento."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
from pathlib import Path   # 👈 faltaba

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.io import ensure_dir, save_lines

def fit_tfidf(
    docs: List[str],
    max_features: int = 2000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int | float = 1,
    max_df: int | float = 0.95,
) -> tuple[TfidfVectorizer, csr_matrix]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,  # ya vienen normalizados
        token_pattern=r"(?u)\b\w+\b",
    )
    X = vectorizer.fit_transform(docs)
    return vectorizer, X

def save_tfidf(X: csr_matrix, path: str) -> None:
    ensure_dir(Path(path).parent)
    save_npz(path, X)

def vocab_to_file(vectorizer: TfidfVectorizer, path: str) -> None:
    vocab = sorted(vectorizer.vocabulary_.keys())
    save_lines(vocab, path)

def top_terms_by_doc(
    X: csr_matrix, vectorizer: TfidfVectorizer, top_k: int = 10
) -> List[List[tuple[str, float]]]:
    feature_names = np.array(sorted(vectorizer.vocabulary_.items(), key=lambda kv: kv[1]))
    terms = feature_names[:, 0]
    top_per_doc: List[List[tuple[str, float]]] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            top_per_doc.append([])
            continue
        indices = row.indices
        data = row.data
        order = np.argsort(-data)[:top_k]
        top_terms = [(terms[indices[j]], float(data[j])) for j in order]
        top_per_doc.append(top_terms)
    return top_per_doc
