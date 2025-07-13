from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import time


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import time
import numpy as np


def entrenar_modelo_base(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo base con RandomForest sin optimización de hiperparámetros.
    Compatible con clasificación binaria y multiclase.
    """
    print("\n🌲 Entrenando modelo base...")

    start = time.time()
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)

    # Intentar obtener probabilidades
    try:
        y_prob = model.predict_proba(X_test)

        if y_prob.ndim == 2 and y_prob.shape[1] > 1:
            # Multiclase
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
            auc = 0.0
        else:
            # Binaria
            auc = roc_auc_score(y_test, y_prob[:, 1])
    except Exception as e:
        print(f"⚠️ No se pudo calcular AUC: {e}")
        y_prob = None
        auc = 0.0

    # Métricas multiclase
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    print("\n📊 Evaluación del modelo base:")
    print(classification_report(y_test, y_pred))
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"⏱️ Tiempo de entrenamiento: {end - start:.2f} segundos")

    return {
        "metodo": "Base",
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "modelo": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "tiempo": end - start,
        "mejores_parametros": None,
    }



def crear_modelo_random_forest(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
    """Crea un modelo RandomForest con parámetros dados"""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )