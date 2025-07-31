# src/modelos.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from src.visualizador import (
    graficar_matriz_confusion,
    graficar_curva_roc,
    graficar_importancia_variables
)


def entrenar_logistic_regression(X_train, y_train, X_test, y_test, output_dir="outputs"):
    """
    Entrena un modelo de Regresión Logística con regularización (L1, L2),
    evalúa con métricas avanzadas y genera visualizaciones.

    Args:
        X_train, y_train, X_test, y_test: Conjuntos de datos.
        output_dir (str): Carpeta donde guardar resultados y gráficos.

    Returns:
        dict: Diccionario con el mejor modelo y resultados.
    """
    print("🔧 Entrenando Regresión Logística con regularización...")

    param_grid = {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 200],
        "solver": ["liblinear", "saga"],
        "max_iter": [500, 1000, 1500],
    }

    grid = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    resultados = {
        "modelo": "LogisticRegression",
        "best_params": grid.best_params_,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": auc
    }

    # Guardar métricas y matriz de confusión
    os.makedirs(output_dir, exist_ok=True)
    df_resultados = pd.DataFrame([resultados])
    df_resultados.to_csv(os.path.join(output_dir, "logistic_regression_resultados.csv"), index=False)

    pd.DataFrame(cm, index=["<=50K", ">50K"], columns=["Pred <=50K", "Pred >50K"]).to_csv(
        os.path.join(output_dir, "logistic_regression_confusion_matrix.csv")
    )

    # 🔹 Visualizaciones (se delegan a visualizador.py)
    graficar_matriz_confusion(cm, "LogisticRegression", output_dir)
    graficar_curva_roc(best_model, X_test, y_test, "LogisticRegression", output_dir)
    graficar_importancia_variables(best_model, X_train, "LogisticRegression", output_dir)

    print(f"✅ Logistic Regression entrenado. Accuracy={acc:.4f}, AUC={auc:.4f}")
    return {"modelo": best_model, "resultados": resultados}


# src/modelos.py


def entrenar_random_forest(X_train, y_train, X_test, y_test, output_dir="outputs"):
    """
    Entrena un modelo Random Forest con hiperparámetros de regularización.
    Evalúa métricas, guarda resultados y genera visualizaciones.

    Args:
        X_train, y_train, X_test, y_test: Conjuntos de datos.
        output_dir (str): Carpeta para guardar resultados y gráficos.

    Returns:
        dict: Diccionario con el mejor modelo y resultados.
    """
    print("🔧 Entrenando Random Forest...")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    resultados = {
        "modelo": "RandomForest",
        "best_params": grid.best_params_,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": auc
    }

    # Guardar métricas y matriz de confusión
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([resultados]).to_csv(os.path.join(output_dir, "random_forest_resultados.csv"), index=False)
    pd.DataFrame(cm, index=["<=50K", ">50K"], columns=["Pred <=50K", "Pred >50K"]).to_csv(
        os.path.join(output_dir, "random_forest_confusion_matrix.csv")
    )

    # 🔹 Visualizaciones desde visualizador.py
    graficar_matriz_confusion(cm, "RandomForest", output_dir)
    graficar_curva_roc(best_model, X_test, y_test, "RandomForest", output_dir)
    graficar_importancia_variables(best_model, X_train, "RandomForest", output_dir)

    print(f"✅ Random Forest entrenado. Accuracy={acc:.4f}, AUC={auc:.4f}")
    return {"modelo": best_model, "resultados": resultados}
