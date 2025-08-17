# main.py
import numpy as np
from sklearn.decomposition import PCA
from src.utils import cargar_dataset
from src.modelos import (
    aplicar_dbscan,
    aplicar_hdbscan,
    aplicar_umap,
    detectar_anomalias_isolation_forest,
    detectar_anomalias_one_class_svm
)
from src.visualizador import visualizar_clusters
from src.evaluador import evaluar_clusterings
import pandas as pd
import os

def main():
    print("🔹 Iniciando pipeline Clustering Basado en Densidad...")

    # Cargar el dataset de diabetes
    X_scaled, y = cargar_dataset("diabetes", usar_minmax=True)

    X = X_scaled  # o seguir usándolo como X directamente

    # Aplicar PCA para reducir a 2D antes de clustering
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    varianza_total = pca.explained_variance_ratio_.sum()
    print(f"🔍 Varianza explicada por PCA: {varianza_total:.2f}")

    # Crear directorio de salida
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Graficar PCA 2D
    visualizar_clusters(X_pca, y, "PCA 2D - Clustering Basado en Densidad")

    resultados_dbscan = []
    for eps in [0.3, 0.5, 0.7]:
        for min_samples in [3, 5, 10]:
            print(f"📌 Evaluando DBSCAN con eps={eps}, min_samples={min_samples}")
            etiquetas = aplicar_dbscan(X_pca, eps=eps, min_samples=min_samples)
            
            # Generar gráficos
            visualizar_clusters(X_pca, etiquetas, 
                                f"DBSCAN eps={eps}, min_samples={min_samples}")
            
            # Evaluación consolidada
            resultados_dbscan.append(
                evaluar_clusterings(X_pca, etiquetas, "DBSCAN", {"eps": eps, "min_samples": min_samples})
            )

    # Guardar CSV consolidado para DBSCAN
    pd.DataFrame(resultados_dbscan).to_csv(f"{output_dir}/DBSCAN_consolidado.csv", index=False)
    print("✅ Resultados DBSCAN guardados en CSV consolidado")

    # Aplicar UMAP
    X_umap = aplicar_umap(X_scaled, n_components=2)
    visualizar_clusters(X_umap, y, "UMAP Visualization")

    resultados_hdbscan = []
    for min_cluster_size in [5, 10]:
        print(f"📌 Evaluando HDBSCAN con min_cluster_size={min_cluster_size}")
        etiquetas = aplicar_hdbscan(X_scaled, min_cluster_size=min_cluster_size)
        
        # Generar gráficos
        visualizar_clusters(X_scaled.values, etiquetas, 
                            f"HDBSCAN min_cluster_size={min_cluster_size}")
        
        # Evaluación consolidada
        resultados_hdbscan.append(
            evaluar_clusterings(X_scaled, etiquetas, "HDBSCAN", {"min_cluster_size": min_cluster_size})
        )

    # Guardar CSV consolidado para HDBSCAN
    pd.DataFrame(resultados_hdbscan).to_csv(f"{output_dir}/HDBSCAN_consolidado.csv", index=False)
    print("✅ Resultados HDBSCAN guardados en CSV consolidado")

    # Detección de anomalías
    anomalies_iso = detectar_anomalias_isolation_forest(X_scaled)
    anomalies_svm = detectar_anomalias_one_class_svm(X_scaled)

    # Mostrar resultados de anomalías
    X['Anomaly Isolation Forest'] = anomalies_iso
    X['Anomaly One-Class SVM'] = anomalies_svm
    print("Pacientes considerados anómalos por Isolation Forest:")
    print(X[X['Anomaly Isolation Forest'] == -1])
    print("Pacientes considerados anómalos por One-Class SVM:")
    print(X[X['Anomaly One-Class SVM'] == -1])

     # Análisis cruzado
    print("\n=== ANÁLISIS CRUZADO ===")
   
if __name__ == "__main__":
    main()
