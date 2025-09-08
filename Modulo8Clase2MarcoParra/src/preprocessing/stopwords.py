"""Carga y unifica stopwords desde spaCy o NLTK."""
from __future__ import annotations


from typing import Set
from nltk.corpus import stopwords
import spacy


def get_stopwords(language: str = "es", source: str = "spacy") -> Set[str]:
    language = language.lower()
    source = source.lower()

    if source == "spacy":
        model = "es_core_news_sm" if language.startswith("es") else "en_core_web_sm"
        nlp = spacy.load(model)
        return set(nlp.Defaults.stop_words)


    if source == "nltk":
        lang_map = {"es": "spanish", "en": "english"}
        lang = lang_map.get(language[:2], "english")
        return set(stopwords.words(lang))


    raise ValueError("source debe ser 'spacy' o 'nltk'")