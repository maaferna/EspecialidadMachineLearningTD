# src/entrenamiento.py

import time
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_pinball_loss,
    accuracy_score
)

from sklearn.metrics import mean_pinball_loss


from src.modelos import (
    crear_modelo_elasticnet,
    crear_modelo_regresion_cuantil,
    crear_modelo_random_forest,
    crear_modelo_xgboost
)

def entrenar_elasticnet(X_train, y_train, X_test, y_test, n_iter=30):
    """
    Entrena ElasticNet con búsqueda aleatoria y evalúa sobre el conjunto de prueba.

    Args:
        X_train: Variables de entrenamiento.
        y_train: Objetivo de entrenamiento.
        X_test: Variables de prueba.
        y_test: Objetivo de prueba.
        n_iter: Número de iteraciones en la búsqueda aleatoria.

    Returns:
        dict: Resultados de entrenamiento y evaluación.
    """
    param_grid = {
        "alpha": np.logspace(-2, 2, 10),
        "l1_ratio": np.linspace(0.1, 1, 5),
    }

    modelo = crear_modelo_elasticnet()

    search = RandomizedSearchCV(
        modelo,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )

    start = time.time()
    search.fit(X_train, y_train)
    end = time.time()

    mejor_modelo = search.best_estimator_
    y_pred = mejor_modelo.predict(X_test)
    score = mean_squared_error(y_test, y_pred, squared=False)

    print(f"[ElasticNet] Best Params: {search.best_params_}")
    print(f"[ElasticNet] RMSE: {score:.4f}")

    return {
        "modelo": "ElasticNet",
        "score": score,
        "tiempo": end - start,
        "parametros": search.best_params_,
        "y_pred": y_pred,
    }


def entrenar_regresion_cuantil(X_train, y_train, X_test, y_test):
    """
    Entrena QuantileRegressor para distintos cuantiles y alphas, 
    pero retorna solo el mejor alpha por cada quantil.

    Returns:
        list[dict]: Lista de resultados con el mejor modelo por quantil.
    """
    resultados = []
    cuantiles = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    alphas = [0.1, 0.5, 1.0]

    for quantil in cuantiles:
        mejor_score = float("inf")
        mejor_resultado = None

        for alpha in alphas:
            modelo = crear_modelo_regresion_cuantil(percentil=quantil, alpha=alpha)

            inicio = time.time()
            modelo.fit(X_train, y_train)
            fin = time.time()

            pred = modelo.predict(X_test)
            score = mean_pinball_loss(y_test, pred, alpha=quantil)

            print(f"[QuantileRegressor-{quantil}, α={alpha}] Pinball Loss: {score:.4f}")

            if score < mejor_score:
                mejor_score = score
                mejor_resultado = {
                    "modelo": f"QuantileRegressor-{quantil}",
                    "score": score,
                    "tiempo": fin - inicio,
                    "parametros": {"quantil": quantil, "alpha": alpha},
                    "y_pred": pred
                }

        resultados.append(mejor_resultado)

    return resultados


def entrenar_random_forest(X_train, y_train, X_test, y_test, n_iter=30):
    """
    Entrena un modelo RandomForestClassifier con RandomizedSearchCV.

    Returns:
        dict: Resultados de evaluación.
    """
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
    }

    modelo = crear_modelo_random_forest()

    search = RandomizedSearchCV(
        modelo,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="accuracy",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )

    inicio = time.time()
    search.fit(X_train, y_train)
    fin = time.time()

    mejor_modelo = search.best_estimator_
    y_pred = mejor_modelo.predict(X_test)
    y_prob = mejor_modelo.predict_proba(X_test)[:, 1]
    score = accuracy_score(y_test, y_pred)

    print(f"[RandomForest] Accuracy: {score:.4f}")

    return {
        "modelo": "RandomForest",
        "score": score,
        "tiempo": fin - inicio,
        "parametros": search.best_params_,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def entrenar_xgboost(X_train, y_train, X_test, y_test, n_iter=20):
    """
    Entrena un modelo XGBoostClassifier con RandomizedSearchCV.

    Returns:
        dict: Resultados de evaluación.
    """
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }

    modelo = crear_modelo_xgboost()

    search = RandomizedSearchCV(
        modelo,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="accuracy",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )

    inicio = time.time()
    search.fit(X_train, y_train)
    fin = time.time()

    mejor_modelo = search.best_estimator_
    y_pred = mejor_modelo.predict(X_test)
    y_prob = mejor_modelo.predict_proba(X_test)[:, 1]
    score = accuracy_score(y_test, y_pred)

    print(f"[XGBoost] Accuracy: {score:.4f}")

    return {
        "modelo": "XGBoost",
        "score": score,
        "tiempo": fin - inicio,
        "parametros": search.best_params_,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
