import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluar_metricas(modelo, X, y, cv):
    """
    Aplica validación cruzada manual con un modelo y retorna métricas promedio.

    Args:
        modelo: Modelo de clasificación.
        X: Features (array-like o matriz).
        y: Target (array-like).
        cv: Estrategia de validación cruzada (como KFold, StratifiedKFold, etc.).

    Returns:
        dict: Promedios de accuracy, precision, recall y F1-score.
    """
    accs, precs, recs, f1s = [], [], [], []

    for train_idx, test_idx in cv.split(X, y):
        modelo.fit(X[train_idx], y[train_idx])
        y_pred = modelo.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], y_pred))
        precs.append(precision_score(y[test_idx], y_pred))
        recs.append(recall_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred))

    return {
        "accuracy": np.mean(accs),
        "precision": np.mean(precs),
        "recall": np.mean(recs),
        "f1_score": np.mean(f1s)
    }
