# src/evaluador.py

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_pinball_loss,
    accuracy_score,
    confusion_matrix,
    classification_report
)

def evaluar_regresion(y_true, y_pred, metodo="Modelo", loss="rmse", alpha=0.5):
    """
    Evalúa un modelo de regresión usando RMSE o Pinball Loss.

    Args:
        y_true (array-like): Valores reales.
        y_pred (array-like): Predicciones del modelo.
        metodo (str): Nombre del modelo.
        loss (str): Tipo de pérdida ('rmse' o 'pinball').
        alpha (float): Cuantil para Pinball Loss.

    Returns:
        dict: Métrica de evaluación y nombre del modelo.
    """
    if loss == "rmse":
        score = mean_squared_error(y_true, y_pred, squared=False)
    elif loss == "pinball":
        score = mean_pinball_loss(y_true, y_pred, alpha=alpha)
    else:
        raise ValueError("Métrica de pérdida no soportada")

    return {
        "metodo": metodo,
        "score": score,
        "metric": loss,
    }

def evaluar_clasificacion(y_true, y_pred, metodo="Modelo"):
    """
    Evalúa un modelo de clasificación.

    Args:
        y_true (array-like): Etiquetas verdaderas.
        y_pred (array-like): Predicciones del modelo.
        metodo (str): Nombre del modelo.

    Returns:
        dict: Accuracy, matriz de confusión y reporte de clasificación.
    """
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "metodo": metodo,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
