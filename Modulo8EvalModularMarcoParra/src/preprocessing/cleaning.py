"""Limpieza básica de texto (usa 'regex' para \p{...})."""
from __future__ import annotations
import regex as re


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE) # URLs http(s):// o www.
EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,}\b") # Emails básicos
NUM_RE = re.compile(r"\b\d+[.,\d]*\b") # Números (enteros y decimales)
PUNCT_SYM_RE = re.compile(r"[\p{P}\p{S}]+", flags=re.UNICODE)  # Puntuación y símbolos




def basic_clean(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_numbers: bool = True,
    remove_punct: bool = True,
    ) -> str:
    """Limpieza básica de texto.
    Parameters
    ----------
    text : str
        Texto a limpiar.
    lowercase : bool
        Si True, pasa a minúsculas.
    remove_urls : bool
        Si True, elimina URLs.
    remove_emails : bool
        Si True, elimina emails.
    remove_numbers : bool
        Si True, elimina números.
    remove_punct : bool
        Si True, elimina puntuación y símbolos.
    Returns
    -------
    str
        Texto limpio.
    """
    if lowercase:
        text = text.lower()
    if remove_urls:
        text = URL_RE.sub(" ", text)
    if remove_emails:
        text = EMAIL_RE.sub(" ", text)
    if remove_numbers:
        text = NUM_RE.sub(" ", text)
    if remove_punct:
        text = PUNCT_SYM_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()