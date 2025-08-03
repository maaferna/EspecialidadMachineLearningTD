from scipy.cluster.hierarchy import linkage, fcluster

def clustering_jerarquico(X, metodo="ward", n_clusters=2):
    """
    Aplica clustering jerárquico aglomerativo.
    Args:
        X (array-like): Datos a agrupar.
        metodo (str): Método de enlace a utilizar. Opciones: 'ward', 'single', 'complete', 'average'.
        n_clusters (int): Número de clusters a formar.
    Returns:
        Z (ndarray): Matriz de enlace generada por el clustering jerárquico.
        etiquetas (ndarray): Etiquetas de los clusters asignadas a cada muestra.
    """
    Z = linkage(X, method=metodo)
    etiquetas = fcluster(Z, n_clusters, criterion="maxclust")
    return Z, etiquetas
