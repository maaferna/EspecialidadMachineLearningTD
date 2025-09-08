"""Limpieza básica de texto."""
from __future__ import annotations

# 👇 usar el motor 'regex', que sí soporta \p{...}
import regex as re

from typing import Optional

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NUM_RE = re.compile(r"\b\d+[.,\d]*\b")

# Puntuación y símbolos Unicode (grupos P y S)
PUNCT_SYM_RE = re.compile(r"[\p{P}\p{S}]+", flags=re.UNICODE)

def basic_clean(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_numbers: bool = True,
    remove_punct: bool = True,
) -> str:
    """Aplica transformaciones básicas de limpieza al texto."""
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
    # Normaliza espacios
    text = re.sub(r"\s+", " ", text).strip()
    return text
