# src/validacion.py

import numpy as np
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    TimeSeriesSplit
)

def obtener_kfold(n_splits=5, random_state=42):
    """Devuelve un validador K-Fold con shuffle."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def obtener_loocv(X, y, n_samples=1000):
    """Devuelve un validador Leave-One-Out en un subconjunto de datos.
    
    Args:
        X (array-like): Características del conjunto de datos.
        y (array-like): Etiquetas del conjunto de datos.
        n_samples (int): Número de muestras a usar para la validación.
    Returns:
        LeaveOneOut: Un objeto LeaveOneOut configurado para el subconjunto.
    """
    # Seleccionar un subconjunto aleatorio de muestras
    if n_samples > len(X):
        n_samples = len(X)  # Asegurarse de no exceder el tamaño del conjunto de datos
    indices = np.random.choice(len(X), size=n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    # Crear el validador Leave-One-Out para el subconjunto
    loo = LeaveOneOut()
    # Retornar el validador y el subconjunto
    return loo, X_subset, y_subset

def obtener_stratified_kfold(n_splits=5, random_state=42):
    """Devuelve un validador Stratified K-Fold."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def obtener_timeseries_split(n_splits=5):
    """Devuelve un validador para series de tiempo."""
    return TimeSeriesSplit(n_splits=n_splits)
