"""Tokenización/lematización con spaCy con fallback si no hay modelo."""
from __future__ import annotations
from typing import List, Sequence, Set
import spacy


class SpacyTokenizer:
    """Tokenizador y lematizador con spaCy. Si no hay modelo, usa tokenización básica.
    - Filtra puntuación, dígitos y espacios.
    - Si hay modelo, filtra por POS (por defecto, NOUN, VERB, ADJ, ADV, PROPN).
    - Convierte a minúsculas.
    - Elimina stopwords (si se proporcionan).
    Parameters
    ----------
    language : str
        Código de idioma (ej. 'es', 'en').
    model_es : str
        Nombre del modelo spaCy para español.
    model_en : str
        Nombre del modelo spaCy para inglés.
    keep_pos : Sequence[str]
        POS a mantener (si hay modelo). Por defecto, NOUN, VERB, ADJ, ADV, PROPN.
    stopwords : Set[str] | None
        Conjunto de stopwords a eliminar (en minúsculas). Si None, no elimina
    
        stopwords.
    """
    def __init__(
                self,
                language: str = "es",
                model_es: str = "es_core_news_sm",
                model_en: str = "en_core_web_sm",
                keep_pos: Sequence[str] = ("NOUN","VERB","ADJ","ADV","PROPN"),
                stopwords: Set[str] | None = None,
                ) -> None:
        
        """
        Inicializa el tokenizador.
        Parameters
        ----------
        language : str
            Código de idioma (ej. 'es', 'en').
        model_es : str
            Nombre del modelo spaCy para español.
        model_en : str
            Nombre del modelo spaCy para inglés.
        keep_pos : Sequence[str]
            POS a mantener (si hay modelo). Por defecto, NOUN, VERB, ADJ, ADV, PROPN.
        stopwords : Set[str] | None
            Conjunto de stopwords a eliminar (en minúsculas). Si None, no elimina
            stopwords.
        Returns
        -------
                """
        self.lang = language.lower()
        model = model_es if self.lang.startswith("es") else model_en
        try:
            self.nlp = spacy.load(model, disable=["ner"]) # Deshabilita NER para acelerar, ner (Named Entity Recognition = reconocimiento de entidades)
            self.has_tagger = True # Si hay modelo, tiene POS y lemas
        except OSError:
            self.nlp = spacy.blank("es" if self.lang.startswith("es") else "en")
            self.has_tagger = False
        self.keep_pos = set(keep_pos) # Por defecto, NOUN, VERB, ADJ, ADV, PROPN
        self.stopwords = {s.lower() for s in (stopwords or set())}


    def __call__(self, text: str) -> List[str]:
        """Tokeniza y lematiza un texto.
        Parameters
        ----------
        text : str
            Texto a procesar.
        Returns
        -------
        List[str]
            Lista de tokens procesados.
        """
        doc = self.nlp(text)
        out: List[str] = []
        for tok in doc:
            if tok.is_space or tok.is_punct or tok.is_digit:
                continue
            if self.has_tagger and self.keep_pos and tok.pos_ not in self.keep_pos:
                continue
            term = (tok.lemma_ if self.has_tagger and tok.lemma_ else tok.text).strip().lower() # Usa lema si hay modelo y lema, sino token original
            if not term or term in self.stopwords:
                continue
            out.append(term)
        return out