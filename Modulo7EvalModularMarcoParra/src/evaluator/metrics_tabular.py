# src/evaluator/metrics_tabular.py
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve
)


def evaluate_classification(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calcula métricas de clasificación binaria y devuelve todo en tipos nativos de Python.
    - y_true: 0/1
    - y_proba: probas en [0,1]
    - threshold: umbral de decisión para clase positiva
    """
    y_pred = (y_proba >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = float(roc_auc_score(y_true, y_proba))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)

    fpr, tpr, thr = roc_curve(y_true, y_proba)

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(), 
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thr": [float(x) for x in thr],
        },
    }


def cost_analysis(cm: np.ndarray, cost_fp: float = 1.0, cost_fn: float = 5.0) -> Dict[str, float]:
    """
    Retorna conteos de la matriz y costo esperado con tipos nativos.
    cm = [[TN, FP], [FN, TP]]
    """
    cm = np.asarray(cm).astype(int)
    TN, FP, FN, TP = cm.ravel()
    return {
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "TP": int(TP),
        "cost_fp": float(cost_fp),
        "cost_fn": float(cost_fn),
        "expected_cost": float(FP) * float(cost_fp) + float(FN) * float(cost_fn),
    }
