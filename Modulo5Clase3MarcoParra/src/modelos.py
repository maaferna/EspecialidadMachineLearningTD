# src/modelos.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def entrenar_modelo(modelo, X_train, y_train, X_test, y_test, grid_params, nombre_modelo):
    """    Entrena un modelo de clasificación con los parámetros especificados y evalúa su rendimiento.

    Args:
        modelo: Modelo de clasificación a entrenar.
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Variable objetivo de prueba.
        grid_params (dict): Parámetros para la búsqueda en cuadrícula.
        nombre_modelo (str): Nombre del modelo para imprimir resultados.

    Returns:
        best_model: El mejor modelo entrenado.
    """
    print(f"🔧 Entrenando modelo {nombre_modelo}...")
    print(f"🔧 Entrenando modelo {nombre_modelo}...")
    grid = GridSearchCV(modelo, grid_params, cv=5, scoring="accuracy", n_jobs=-1, return_train_score=False)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"✅ {nombre_modelo} Accuracy: {acc:.4f}")
    print(f"📊 Matriz de Confusión ({nombre_modelo}):\n{cm}\n")

    # 🔹 Guardar resultados detallados de todas las combinaciones
    df_all = pd.DataFrame(grid.cv_results_)
    df_all["model"] = nombre_modelo
    df_all[["model", "params", "mean_test_score"]].to_csv(f"outputs/{nombre_modelo.lower()}_all_results.csv", index=False)

    # 🔹 Guardar mejor resultado
    resultados = pd.DataFrame({
        "model": [nombre_modelo],
        "accuracy": [acc],
        "best_params": [grid.best_params_]
    })
    resultados.to_csv(f"outputs/{nombre_modelo.lower()}_mejor.csv", index=False)

    # 🔹 Guardar matriz de confusión como CSV
    cm_df = pd.DataFrame(cm, index=["Real_0", "Real_1"], columns=["Pred_0", "Pred_1"])
    cm_df.to_csv(f"outputs/matriz_confusion_{nombre_modelo.lower()}.csv")

    return best_model

def entrenar_random_forest(X_train, y_train, X_test, y_test):
    """Entrena un modelo Random Forest y evalúa su rendimiento.

    Args:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns:
        best_model: El mejor modelo Random Forest entrenado.
    """
    params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
    return entrenar_modelo(RandomForestClassifier(), X_train, y_train, X_test, y_test, params, "RandomForest")

def entrenar_adaboost(X_train, y_train, X_test, y_test):
    """Entrena un modelo AdaBoost y evalúa su rendimiento.

    Args:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns:
        best_model: El mejor modelo AdaBoost entrenado.
    """
    params = {
        "n_estimators": [50, 100],
        "learning_rate": [0.5, 1.0]
    }
    return entrenar_modelo(AdaBoostClassifier(), X_train, y_train, X_test, y_test, params, "AdaBoost")

def entrenar_xgboost(X_train, y_train, X_test, y_test):
    """Entrena un modelo XGBoost y evalúa su rendimiento.
    Args:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns:
        best_model: El mejor modelo XGBoost entrenado.
    """
    params = {
        "n_estimators": [100, 200],
        "learning_rate": [0.1, 0.3],
        "max_depth": [3, 6]
    }
    return entrenar_modelo(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), X_train, y_train, X_test, y_test, params, "XGBoost")
   

def entrenar_lightgbm(X_train, y_train, X_test, y_test):
    """Entrena un modelo LightGBM y evalúa su rendimiento.
    Args:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns:
        best_model: El mejor modelo LightGBM entrenado.
    """
    params = {
        "n_estimators": [100],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31]
    }
    return entrenar_modelo(LGBMClassifier(force_col_wise=True, verbose=-1), X_train, y_train, X_test, y_test, params, "LightGBM")

def entrenar_catboost(X_train, y_train, X_test, y_test):
    """Entrena un modelo CatBoost y evalúa su rendimiento.
    Args:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns:
        best_model: El mejor modelo CatBoost entrenado.
    """
    # CatBoost no requiere preprocesamiento de datos categóricos
    params = {
        "iterations": [100, 200],
        "learning_rate": [0.05, 0.1],
        "depth": [4, 6]
    }
    return entrenar_modelo(CatBoostClassifier(verbose=0), X_train, y_train, X_test, y_test, params, "CatBoost")
