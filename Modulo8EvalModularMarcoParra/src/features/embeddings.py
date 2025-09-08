# src/features/embeddings.py
from __future__ import annotations
from typing import Iterable, List
import numpy as np
from gensim.models import Word2Vec, FastText

def train_word2vec(token_seqs: Iterable[List[str]], size: int = 100, window: int = 5, min_count: int = 2, sg: int = 1, workers: int = 2):
    """ Entrena modelo Word2Vec (Skip-gram o CBOW) sobre secuencias de tokens."""
    model = Word2Vec(sentences=list(token_seqs), vector_size=size, window=window, min_count=min_count, sg=sg, workers=workers)
    return model

def train_fasttext(token_seqs: Iterable[List[str]], size: int = 100, window: int = 5, min_count: int = 2, sg: int = 1, workers: int = 2):
    """ Entrena modelo FastText (Skip-gram o CBOW) sobre secuencias de tokens."""
    model = FastText(sentences=list(token_seqs), vector_size=size, window=window, min_count=min_count, sg=sg, workers=workers)
    return model

def doc_mean_vector(tokens: List[str], model, normalize: bool = False) -> np.ndarray:
    """Promedio simple de embeddings de tokens presentes en el vocab.
    Si ningún token está en el vocabulario, devuelve vector cero.
    Si normalize=True, normaliza el vector resultante a norma L2=1."""
    vecs = []
    for t in tokens:
        if t in model.wv:
            vecs.append(model.wv[t])
    if not vecs:
        # Si ningún token está en vocabulario, devuelve vector cero
        return np.zeros(model.wv.vector_size, dtype=np.float32)
    M = np.vstack(vecs).mean(axis=0)
    if normalize:
        n = np.linalg.norm(M) + 1e-12
        M = M / n
    return M

def docs_to_matrix(token_seqs: Iterable[List[str]], model, normalize: bool = False) -> np.ndarray:
    """Convierte secuencia de tokens a matriz 2D de embeddings promedio por documento."""
    return np.vstack([doc_mean_vector(tokens, model, normalize=normalize) for tokens in token_seqs])
