from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def crear_modelo_random_forest(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="auto",
    random_state=42,
    n_jobs=-1
):
    """
    Crea y retorna un modelo RandomForestClassifier con parámetros configurables.

    Args:
        n_estimators (int): Número de árboles en el bosque.
        max_depth (int or None): Profundidad máxima del árbol.
        min_samples_split (int): Mínimo número de muestras para dividir un nodo.
        min_samples_leaf (int): Mínimo número de muestras en una hoja.
        max_features (str or int): Número de características a considerar.
        random_state (int): Semilla para reproducibilidad.
        n_jobs (int): Número de trabajos paralelos.

    Returns:
        RandomForestClassifier: Modelo inicializado.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=n_jobs
    )


def crear_modelo_logistic_regression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=42
):
    """
    Crea y retorna un modelo LogisticRegression con hiperparámetros configurables.

    Args:
        penalty (str): Tipo de penalización ("l1", "l2", etc.).
        C (float): Inverso de la regularización.
        solver (str): Algoritmo de optimización.
        max_iter (int): Máximo número de iteraciones.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        LogisticRegression: Modelo inicializado.
    """
    return LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )


def crear_modelo_svc(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    probability=True,
    random_state=42
):
    """
    Crea y retorna un modelo SVC con parámetros configurables.

    Args:
        C (float): Parámetro de penalización.
        kernel (str): Tipo de kernel a usar ("linear", "poly", "rbf", etc.).
        gamma (str): Coeficiente del kernel.
        probability (bool): Si se debe habilitar la estimación de probabilidad.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        SVC: Modelo inicializado.
    """
    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        random_state=random_state
    )
