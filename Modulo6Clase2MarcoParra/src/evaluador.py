import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluar_clusterings(X, etiquetas, modelo, params, output_dir="outputs"):
    """
    Evalúa un clustering y devuelve resultados en forma de diccionario.

    Args:
        X (array-like): Datos de entrada.
        etiquetas (array-like): Etiquetas asignadas por el clustering.
        modelo (str): Nombre del modelo (DBSCAN o HDBSCAN).
        params (dict): Diccionario con los parámetros usados.
        output_dir (str): Carpeta donde guardar los resultados.

    Returns:
        dict: Resultados de métricas y parámetros.
    """
    # Filtrar clusters válidos (ignorar ruido con etiqueta -1)
    if len(set(etiquetas)) <= 1 or (set(etiquetas) == {-1}):
        silhouette = None
        davies = None
    else:
        silhouette = silhouette_score(X, etiquetas)
        davies = davies_bouldin_score(X, etiquetas)

    return {
        "modelo": modelo,
        **params,
        "n_clusters": len(set(etiquetas)) - (1 if -1 in etiquetas else 0),
        "silhouette": silhouette,
        "davies_bouldin": davies
    }
