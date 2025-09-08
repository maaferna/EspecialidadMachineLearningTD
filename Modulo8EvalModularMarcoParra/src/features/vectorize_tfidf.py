"""TF-IDF vectorizer y utilidades."""
from __future__ import annotations
from typing import Tuple, List
from pathlib import Path
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.io import ensure_dir




def fit_tfidf(
    docs: List[str],
    max_features: int = 5000,
    ngram_range: Tuple[int,int] = (1,2),
    min_df: int | float = 1,
    max_df: int | float = 0.95,
    ) -> tuple[TfidfVectorizer, csr_matrix]:
    """Ajusta un vectorizador TF-IDF a `docs` y devuelve el vectorizador y la matriz dispersa.
    Parameters
    ----------
    docs : List[str]
        Documentos de texto.
    max_features : int
        Número máximo de características (términos).
    ngram_range : Tuple[int,int]
        Rango de n-gramas (min, max).
    min_df : int | float
        Frecuencia documental mínima (entero o proporción).
    max_df : int | float
        Frecuencia documental máxima (entero o proporción)."""
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        ) # token_pattern para que tome tokens de una sola letra
    X = vec.fit_transform(docs) # sparse matrix
    return vec, X




def save_sparse(X: csr_matrix, path: str | Path) -> None:
    """Guarda la matriz dispersa `X` en `path` (formato .npz)."""
    ensure_dir(Path(path).parent)
    save_npz(path, X)