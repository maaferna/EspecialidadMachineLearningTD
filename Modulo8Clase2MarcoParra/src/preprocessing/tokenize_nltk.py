"""Tokenización con NLTK y normalización opcional (stem/lemma)."""
from __future__ import annotations


from typing import List


import nltk




def nltk_tokens(
text: str,
language: str = "es",
normalize: str = "stem", # 'none' | 'lemma' | 'stem'
stopwords: set[str] | None = None,
) -> List[str]:
    lang_map = {"es": "spanish", "en": "english"}
    lang = lang_map.get(language[:2], "english")


    tokens = nltk.word_tokenize(text, language=lang)


    if stopwords:
        tokens = [t for t in tokens if t.lower() not in stopwords]


    if normalize == "lemma":
        # Nota: lematización de NLTK funciona bien para inglés (WordNetLemmatizer).
        # Para español, suele ser limitada; se sugiere 'stem'.
        lemmatizer = nltk.stem.WordNetLemmatizer()
        # Asumimos 'n' por simplicidad; para mejor calidad mapear POS.
        tokens = [lemmatizer.lemmatize(t.lower(), pos="n") for t in tokens]
    elif normalize == "stem":
        stemmer = nltk.stem.SnowballStemmer("spanish" if language.startswith("es") else "english")
        tokens = [stemmer.stem(t.lower()) for t in tokens]
    else:
        tokens = [t.lower() for t in tokens]


    # Filtra tokens no alfanuméricos residuales
    tokens = [t for t in tokens if any(ch.isalnum() for ch in t)]
    return tokens