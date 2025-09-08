"""Métricas de clasificación y matplotlib helpers."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    )




def summary_metrics(y_true, y_pred, labels: list[str]) -> Dict:
    """Calcula métricas comunes y devuelve un resumen en un diccionario.
    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_pred : np.ndarray
        Etiquetas predichas.
    labels : list[str]
        Lista de etiquetas (nombres o índices) para el informe y la matriz de confusión.
    Returns
    -------
    Dict
        Diccionario con las métricas calculadas:
        - accuracy: Exactitud global.
        - f1_macro: F1-score macro.
        - report: Informe de clasificación detallado (diccionario).
        - confusion: Matriz de confusión (lista de listas)."""
    
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0) # zero_division=0 para evitar NaN
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1_macro": f1m, "report": report, "confusion": cm.tolist()}