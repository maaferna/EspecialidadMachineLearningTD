from sklearn.cluster import DBSCAN
import hdbscan

def aplicar_dbscan(X, eps=0.5, min_samples=5):
    """Aplica DBSCAN al dataset.
    defaults a eps=0.5 y min_samples=5.
    Args:
        X (pd.DataFrame): Dataset estandarizado.
        eps (float): Distancia máxima entre dos muestras para que se consideren en el mismo vecindario.
        min_samples (int): Número mínimo de muestras en un vecindario para considerar un punto como núcleo.
    Returns:
        etiquetas (array): Etiquetas de los clusters asignados por DBSCAN.
    """
    modelo = DBSCAN(eps=eps, min_samples=min_samples)
    etiquetas = modelo.fit_predict(X)
    return etiquetas



def aplicar_hdbscan(X, min_cluster_size=5, min_samples=None):
    """Aplica HDBSCAN al dataset.
    defaults a min_cluster_size=5.
    Args:
        X (pd.DataFrame): Dataset estandarizado.
        min_cluster_size (int): Tamaño mínimo de un cluster para ser considerado válido.
    Returns:
        etiquetas (array): Etiquetas de los clusters asignados por HDBSCAN.
    """
    print(f"🔧 Ejecutando HDBSCAN (min_cluster_size={min_cluster_size})...")
    modelo = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    etiquetas = modelo.fit_predict(X)
    return etiquetas
