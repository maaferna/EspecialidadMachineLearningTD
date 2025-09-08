# src/features/transformer_embedder.py
from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

def load_st_model(model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"):
    """Carga modelo SentenceTransformer por nombre."""
    return SentenceTransformer(model_name)

def encode_texts(model, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    """ Codifica lista de textos a matriz 2D de embeddings.
     Si normalize=True, normaliza cada embedding a norma L2=1.
     """
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=normalize)
    return emb
