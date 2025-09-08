"""Tokenización con NLTK (con auto-descarga de recursos) y normalización opcional."""
from __future__ import annotations
from typing import List
import re, nltk




def _ensure_tokenizers(language: str = "es") -> None:
    """Asegura que los tokenizadores necesarios estén descargados.
    Parameters
    ----------
    language : str
        Código de idioma (es/en).
    Tokenizadores son necesarios para nltk.word_tokenize.
    """
    try:
        # Prueba si el tokenizador está disponible, Punkt es el más común permitendo tokenizar en varios idiomas
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try: nltk.download("punkt", quiet=True, raise_on_error=True)
        except Exception: pass
    try:
        nltk.data.find(f"tokenizers/punkt_tab/{'spanish' if language.startswith('es') else 'english'}/")
    except LookupError:
        try: nltk.download("punkt_tab", quiet=True, raise_on_error=True)
        except Exception: pass


def _regex_tokens(text: str) -> List[str]:
    """Tokenización básica por regex (fallback si nltk falla)."""
    return re.findall(r"\w+", text, flags=re.UNICODE)




def nltk_tokenize(
    text: str,
    language: str = "es",
    normalize: str = "stem", # none|lemma|stem
    stopwords: set[str] | None = None,
    ) -> List[str]:
    """
    Tokeniza con NLTK y normaliza (opcional).
    Si falla NLTK, usa tokenización básica por regex.
    Parameters
    ----------
    text : str
        Texto a tokenizar.
    language : str
        Código de idioma (es/en).
    normalize : str
        Tipo de normalización: 'none' (ninguna), 'lemma' (lematización) o 'stem' (stemming).
        Por defecto, 'stem'.
    stopwords : set[str] | None
        Conjunto de stopwords a eliminar (en minúsculas). Si None, no elimina stopwords.
    Returns
    -------"""
    lang_map = {"es":"spanish","en":"english"}
    lang = lang_map.get(language[:2], "english")
    _ensure_tokenizers(language)
    try:
        tokens = nltk.word_tokenize(text, language=lang) # Tokeniza el texto, puede fallar si no hay recursos
    except Exception:
        tokens = _regex_tokens(text)
    if stopwords: # Permite eliminar stopwords
        tokens = [t for t in tokens if t.lower() not in stopwords]
    if normalize == "lemma": # Lematiza (requiere WordNet)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t.lower(), pos="n") for t in tokens]
    elif normalize == "stem": # Stemming (básico, sin POS)
        stemmer = nltk.stem.SnowballStemmer("spanish" if language.startswith("es") else "english")
        tokens = [stemmer.stem(t.lower()) for t in tokens]
    else:
        tokens = [t.lower() for t in tokens]
    return [t for t in tokens if any(ch.isalnum() for ch in t)]