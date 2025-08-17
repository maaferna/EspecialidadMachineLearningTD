# src/analizador.py
import numpy as np
import pandas as pd

def analizar_coincidencias(anomalias_labels, cluster_labels):
    """
    Analiza coincidencias entre anomalías y clústeres raros
    
    Args:
        anomalias_labels (array): Etiquetas de anomalías (-1 = anómalo)
        cluster_labels (array): Etiquetas de clusters
        
    Returns:
        dict: Métricas de coincidencia y ejemplos
    """
    # Identificar clústeres raros (percentil 25 de tamaño)
    _, counts = np.unique(cluster_labels, return_counts=True)
    rare_threshold = np.percentile(counts, 25)
    rare_clusters = [i for i, cnt in enumerate(counts) if cnt < rare_threshold]
    
    # Calcular solapamiento
    anomalies_mask = (anomalias_labels == -1)
    rare_mask = np.isin(cluster_labels, rare_clusters)
    
    total_anomalies = np.sum(anomalies_mask)
    overlap = np.sum(anomalies_mask & rare_mask)
    
    return {
        'porcentaje_coincidencia': (overlap / total_anomalies) * 100 if total_anomalies > 0 else 0,
        'clusters_raros': rare_clusters,
        'ejemplos': None  # Se puede añadir muestras específicas
    }

