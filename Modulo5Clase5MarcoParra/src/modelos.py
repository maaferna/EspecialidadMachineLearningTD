# src/modelos.py

from sklearn.linear_model import Lasso, Ridge, ElasticNet


def crear_modelo_lasso(alpha=1.0, random_state=42):
    """
    Crea un modelo de regresión Lasso (L1).

    Args:
        alpha (float): Parámetro de regularización.

    Returns:
        Lasso: Modelo Lasso.
    """
    return Lasso(alpha=alpha, random_state=random_state)

def crear_modelo_ridge(alpha=1.0, random_state=42):
    """
    Crea un modelo de regresión Ridge (L2).

    Args:
        alpha (float): Parámetro de regularización.

    Returns:
        Ridge: Modelo Ridge.
    """
    return Ridge(alpha=alpha, random_state=random_state)

def crear_modelo_elasticnet(alpha=1.0, l1_ratio=0.5, random_state=42):
    """
    Crea un modelo de regresión ElasticNet (L1 + L2).

    Args:
        alpha (float): Parámetro de regularización.
        l1_ratio (float): Proporción entre L1 y L2.

    Returns:
        ElasticNet: Modelo ElasticNet.
    """
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
