# src/features/text.py

from __future__ import annotations
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer

def simple_clean(s: str, lowercase=True, remove_punct=True, remove_numbers=False):
    """Limpieza simple de texto.
    Args:
        s (str): Texto a limpiar.
        lowercase (bool): Convertir a minúsculas.
        remove_punct (bool): Eliminar puntuación.
        remove_numbers (bool): Eliminar números."""
    if lowercase:
        s = s.lower()
    if remove_punct:
        s = re.sub(r"[^\w\sáéíóúüñ]", " ", s, flags=re.UNICODE)
    if remove_numbers:
        s = re.sub(r"\d+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

@dataclass
class Preprocessor:
    """Clase para preprocesar texto.
    Esta clase es picklable y puede ser usada como preprocesador en TfidfVectorizer.

    """
    lowercase: bool = True
    remove_punct: bool = True
    remove_numbers: bool = False
    def __call__(self, x: str) -> str:
        return simple_clean(
            x,
            lowercase=self.lowercase,
            remove_punct=self.remove_punct,
            remove_numbers=self.remove_numbers,
        )

def build_vectorizer(cfg) -> TfidfVectorizer:
    """Construye un vectorizador TF-IDF basado en la configuración dada.
    Args:
        cfg (dict): Configuración con parámetros de preprocesamiento y vectorización.
    Returns:
        TfidfVectorizer: Vectorizador TF-IDF configurado.
    """
    prep = cfg["preprocessing"]
    stop = prep.get("stopwords", "english")
    ngram = tuple(prep.get("ngram_range", [1, 2]))
    maxf = prep.get("max_features", 20000)

    preproc = Preprocessor(
        lowercase=prep.get("lowercase", True),
        remove_punct=prep.get("remove_punct", True),
        remove_numbers=prep.get("remove_numbers", False),
    )

    return TfidfVectorizer(
        preprocessor=preproc,   # <- ahora es un objeto top-level picklable
        stop_words=stop,
        ngram_range=ngram,
        max_features=maxf
    )

