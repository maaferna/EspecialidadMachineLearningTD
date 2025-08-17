import numpy as np
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import umap.umap_ as umap

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



def detectar_anomalias_isolation_forest(X_scaled):
    """
    Aplica Isolation Forest para detectar anomalías.
    
    Args:
        X_scaled (np.ndarray): Datos escalados.
    
    Returns:
        np.ndarray: Etiquetas de anomalías (-1 para anómalo, 1 para normal).
    """
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X_scaled)
    return anomalies

def detectar_anomalias_one_class_svm(X_scaled):
    """
    Aplica One-Class SVM para detectar anomalías.
    
    Args:
        X_scaled (np.ndarray): Datos escalados.
    
    Returns:
        np.ndarray: Etiquetas de anomalías (-1 para anómalo, 1 para normal).
    """
    oc_svm = OneClassSVM(gamma='auto', nu=0.1)
    anomalies = oc_svm.fit_predict(X_scaled)
    return anomalies


def aplicar_umap(X_scaled, n_components=2):
    """
    Aplica UMAP para la reducción de dimensionalidad.
    
    Args:
        X_scaled (np.ndarray): Datos escalados.
        n_components (int): Número de componentes a reducir.
    
    Returns:
        np.ndarray: Datos reducidos a n componentes.
    """
    umap_model = umap.UMAP(n_components=n_components, random_state=42)
    X_umap = umap_model.fit_transform(X_scaled)
    return X_umap





def entrenar_modelos(X_scaled, if_contamination=0.05, svm_gamma=0.1, svm_nu=0.05):
    """Entrena Isolation Forest y One-Class SVM."""
    iso_model = IsolationForest(contamination=if_contamination, random_state=42)
    iso_model.fit(X_scaled)
    iso_pred = np.where(iso_model.predict(X_scaled) == -1, 1, 0)  # 1 = anómalo

    svm_model = OneClassSVM(kernel="rbf", gamma=svm_gamma, nu=svm_nu)
    svm_model.fit(X_scaled)
    svm_pred = np.where(svm_model.predict(X_scaled) == -1, 1, 0)

    return iso_pred, svm_pred, {"contamination": if_contamination, "gamma": svm_gamma, "nu": svm_nu}
