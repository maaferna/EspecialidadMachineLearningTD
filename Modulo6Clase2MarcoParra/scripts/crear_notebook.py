# create_notebook.py

import nbformat as nbf
from pathlib import Path

def create_notebook():
    # Paths
    notebook_path = Path("notebooks/clustering_pipeline.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a new notebook
    nb = nbf.v4.new_notebook()
    nb["cells"] = []

    # Add a title markdown cell
    nb["cells"].append(nbf.v4.new_markdown_cell("# 📊 Pipeline de Clustering Basado en Densidad"))

    # Add introduction
    nb["cells"].append(nbf.v4.new_markdown_cell("""
## 🎯 Introducción

Este notebook ejecuta un pipeline de clustering basado en densidad utilizando **DBSCAN** y **HDBSCAN** sobre el dataset *Wine*.
"""))

    # Add environment setup cell
    nb["cells"].append(nbf.v4.new_code_cell("""
# ✅ Configuración del entorno
import pandas as pd
import os
from sklearn.decomposition import PCA
from src.utils import cargar_dataset
from src.modelos import aplicar_dbscan, aplicar_hdbscan
from src.visualizador import graficar_clusters, graficar_tsne
from src.evaluador import evaluar_clusterings
"""))

    # Add main pipeline code
    nb["cells"].append(nbf.v4.new_code_cell("""
# 🚀 Ejecutar pipeline principal
print("🔹 Iniciando pipeline Clustering Basado en Densidad...")

X, y = cargar_dataset("wine", usar_minmax=True)
# Aplicar PCA para reducir a 2D antes de clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

varianza_total = pca.explained_variance_ratio_.sum()
print(f"🔍 Varianza explicada por PCA: {varianza_total:.2f}")

# Graficar PCA 2D
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
graficar_clusters(pd.DataFrame(X_pca), y, "PCA 2D - Clustering Basado en Densidad", "outputs/pca_2d.png")

resultados_dbscan = []
for eps in [0.3, 0.5, 0.7]:
    for min_samples in [3, 5, 10]:
        print(f"📌 Evaluando DBSCAN con eps={eps}, min_samples={min_samples}")
        etiquetas = aplicar_dbscan(X_pca, eps=eps, min_samples=min_samples)
        
        # Generar gráficos
        graficar_clusters(X, etiquetas, 
                          f"DBSCAN eps={eps}, min_samples={min_samples}", 
                          os.path.join(output_dir, f"dbscan_eps{eps}_min{min_samples}.png"))
        graficar_tsne(X, etiquetas, 
                      f"t-SNE DBSCAN eps={eps}, min_samples={min_samples}", 
                      os.path.join(output_dir, f"tsne_dbscan_eps{eps}_min{min_samples}.png"))
        
        # Evaluación consolidada
        resultados_dbscan.append(
            evaluar_clusterings(X, etiquetas, "DBSCAN", {"eps": eps, "min_samples": min_samples})
        )

# Guardar CSV consolidado para DBSCAN
pd.DataFrame(resultados_dbscan).to_csv(f"{output_dir}/DBSCAN_consolidado.csv", index=False)
print("✅ Resultados DBSCAN guardados en CSV consolidado")

resultados_hdbscan = []
for min_cluster_size in [5, 10]:
    print(f"📌 Evaluando HDBSCAN con min_cluster_size={min_cluster_size}")
    etiquetas = aplicar_hdbscan(X_pca, min_cluster_size=min_cluster_size)
    
    # Generar gráficos
    graficar_clusters(X, etiquetas, 
                      f"HDBSCAN min_cluster_size={min_cluster_size}", 
                      os.path.join(output_dir, f"hdbscan_mincluster{min_cluster_size}.png"))
    graficar_tsne(X, etiquetas, 
                  f"t-SNE HDBSCAN min_cluster_size={min_cluster_size}", 
                  os.path.join(output_dir, f"tsne_hdbscan_mincluster{min_cluster_size}.png"))
    
    # Evaluación consolidada
    resultados_hdbscan.append(
        evaluar_clusterings(X_pca, etiquetas, "HDBSCAN", {"min_cluster_size": min_cluster_size})
    )

# Guardar CSV consolidado para HDBSCAN
pd.DataFrame(resultados_hdbscan).to_csv(f"{output_dir}/HDBSCAN_consolidado.csv", index=False)
print("✅ Resultados HDBSCAN guardados en CSV consolidado")
"""))

    # Save notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"✅ Notebook creado en: {notebook_path}")

if __name__ == "__main__":
    create_notebook()
