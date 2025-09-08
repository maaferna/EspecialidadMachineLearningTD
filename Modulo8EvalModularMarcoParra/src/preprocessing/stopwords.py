"""Stopwords de spaCy o NLTK con fallback."""
from __future__ import annotations
from typing import Set




def get_stopwords(language: str = "es", source: str = "spacy") -> Set[str]:
    language = language.lower()
    source = source.lower()
    if source == "spacy":
        import spacy
        model = "es_core_news_sm" if language.startswith("es") else "en_core_web_sm"
        try:
            nlp = spacy.load(model)
        except OSError:
            nlp = spacy.blank("es" if language.startswith("es") else "en")
        return set(nlp.Defaults.stop_words)
    if source == "nltk":
        from nltk.corpus import stopwords
        lang_map = {"es": "spanish", "en": "english"}
        return set(stopwords.words(lang_map.get(language[:2], "english")))
    raise ValueError("source debe ser 'spacy' o 'nltk'")