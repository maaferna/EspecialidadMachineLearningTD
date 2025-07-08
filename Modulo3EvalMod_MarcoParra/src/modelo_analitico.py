import numpy as np
from src.excepciones import MatrizSingularError

def calculo_cerrado_regresion(x, y):
    """
    Aplica la fórmula cerrada para regresión lineal: β = (Xᵀ X)^(-1) Xᵀ y

    Parámetros:
    - x: vector de entradas
    - y: vector de salidas

    Retorna:
    - w: pendiente
    - b: intercepto
    """
    try:
        # 1. Construcción de la matriz de diseño con columna de unos
        X_matrix = np.c_[np.ones_like(x), x]  # shape: (n_samples, 2)

        # 2. Cálculo de parámetros beta = (XᵀX)^-1 Xᵀ y
        XTX = X_matrix.T @ X_matrix
        XTy = X_matrix.T @ y
        beta = np.linalg.inv(XTX) @ XTy  # beta = [b, w]

        b, w = beta[0], beta[1]
        return w, b

    except np.linalg.LinAlgError:
        raise MatrizSingularError()
