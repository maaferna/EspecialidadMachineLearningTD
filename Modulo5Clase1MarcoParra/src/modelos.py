# src/modelos.py

from sklearn.linear_model import ElasticNet, QuantileRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# -----------------------------
# MODELOS DE REGRESIÓN
# -----------------------------

def crear_modelo_elasticnet(alpha=1.0, l1_ratio=0.5, random_state=42):
    """
    Crea un modelo ElasticNet.

    Args:
        alpha (float): Intensidad de regularización.
        l1_ratio (float): Proporción de L1 vs L2.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        ElasticNet: Modelo instanciado.
    """
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)


def crear_modelo_regresion_cuantil(percentil=0.5, alpha=1.0, solver="highs"):
    """
    Crea un modelo de Regresión Cuantílica.

    Args:
        percentil (float): Cuantil a estimar (ej. 0.1, 0.5, 0.9).
        alpha (float): Regularización.
        solver (str): Algoritmo de optimización.

    Returns:
        QuantileRegressor: Modelo instanciado.
    """
    return QuantileRegressor(quantile=percentil, alpha=alpha, solver=solver)


# -----------------------------
# MODELOS DE CLASIFICACIÓN
# -----------------------------

def crear_modelo_random_forest(
    n_estimators=100, max_depth=None, min_samples_split=2, random_state=42
):
    """
    Crea un modelo RandomForestClassifier.

    Args:
        n_estimators (int): Nº de árboles.
        max_depth (int or None): Profundidad máxima.
        min_samples_split (int): Mínimo de muestras para dividir un nodo.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        RandomForestClassifier: Modelo instanciado.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )


def crear_modelo_xgboost(
    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
):
    """
    Crea un modelo XGBoostClassifier.

    Args:
        n_estimators (int): Nº de árboles.
        max_depth (int): Profundidad máxima de cada árbol.
        learning_rate (float): Tasa de aprendizaje.
        random_state (int): Semilla.

    Returns:
        XGBClassifier: Modelo instanciado.
    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="logloss",
        random_state=random_state,
    )
