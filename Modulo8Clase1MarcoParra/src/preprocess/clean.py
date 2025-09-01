from __future__ import annotations
import re
from typing import List

_PUNCT_RE = re.compile(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]")

def basic_clean(doc: str) -> str:
    """
    Limpieza mínima (ES):
    - minúsculas
    - elimina puntuación (mantiene tildes y ñ)
    - colapsa espacios
    """
    doc = doc.lower()
    doc = _PUNCT_RE.sub(" ", doc)
    doc = re.sub(r"\s+", " ", doc).strip()
    return doc

def clean_corpus(docs: List[str]) -> List[str]:
    """Aplica basic_clean a cada documento."""
    return [basic_clean(d) for d in docs]
