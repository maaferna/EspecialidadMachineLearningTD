"""Predicción con artefactos guardados."""
from __future__ import annotations
from typing import List, Tuple
import joblib # Para cargar modelos y artefactos




def load_artifacts(vec_path: str, model_path: str, le_path: str):
    """Carga vectorizador, modelo y codificador de etiquetas desde archivos."""
    vec = joblib.load(vec_path)
    clf = joblib.load(model_path)
    le = joblib.load(le_path)
    return vec, clf, le




def predict_texts(texts: List[str], preprocess_fn, vec, clf, le) -> Tuple[list[str], list[int]]:
    """Predice etiquetas para una lista de textos.
    Parameters
    ----------
    texts : List[str]
        Lista de textos a predecir.
    preprocess_fn : Callable[[str], str]
        Función de preprocesamiento de texto.
    vec : VectorizerMixin
        Vectorizador (entrenado).
    clf : ClassifierMixin
        Clasificador (entrenado).
    le : LabelEncoder
        Codificador de etiquetas (entrenado).
    Returns
    -------
    Tuple[List[str], List[int]]
        Etiquetas predichas (str) y sus índices (int).
    -------

    """
    X = [preprocess_fn(t) for t in texts]
    Xv = vec.transform(X)
    y_pred = clf.predict(Xv)
    labels = le.inverse_transform(y_pred) # Permite obtener etiquetas originales
    return labels, y_pred.tolist()