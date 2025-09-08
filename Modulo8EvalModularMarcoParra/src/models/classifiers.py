# src/models/classifiers.py
from __future__ import annotations
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def build_classifier(name: str, cfg: dict):
    """Crea clasificador según nombre y parámetros en cfg.
    name: multinomial_nb | logreg | linear_svm
    cfg: diccionario con parámetros específicos del modelo.
    
    """
    name = name.lower()
    if name == "multinomial_nb":
        alpha = cfg["model"].get("alpha", 1.0)
        return MultinomialNB(alpha=alpha)
    if name == "logreg":
        C = cfg["model"].get("C", 1.0)
        return LogisticRegression(max_iter=2000, C=C, class_weight=cfg["model"].get("class_weight", None))
    if name == "linear_svm":
        C = cfg["model"].get("C", 1.0)
        return LinearSVC(C=C, class_weight=cfg["model"].get("class_weight", None))
    raise ValueError(f"Modelo no soportado: {name}")
