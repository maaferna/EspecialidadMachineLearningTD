# create_notebooks.py
import nbformat as nbf
from pathlib import Path
from textwrap import dedent

def md(s):  return nbf.v4.new_markdown_cell(dedent(s).strip())
def code(s): return nbf.v4.new_code_cell(dedent(s).strip())

def make_nb_01():
    nb = nbf.v4.new_notebook()
    nb.cells = []

    # Título + propósito
    nb.cells += [
        md("""
        # 📊 Pipeline de Clustering Basado en Densidad (Diabetes)
        Equivalente a `scripts/main.py`. Ejecuta:
        - Carga y escalado del dataset de **diabetes** (Kaggle).
        - **PCA (2D)** y visualización.
        - **DBSCAN** (varios `eps`/`min_samples`) + métricas (Silueta, Davies–Bouldin).
        - **UMAP** (2D) si está disponible.
        - **HDBSCAN** con distintos `min_cluster_size`.
        - **Detección de anomalías** con Isolation Forest y One-Class SVM.
        """),
        md("""
        ## 🔧 Requisitos
        Asegúrate de tener el entorno activo e instalar dependencias opcionales:
        ```bash
        conda activate especialidadmachinelearning
        pip install umap-learn hdbscan kagglehub
        ```
        Si quieres forzar una ruta local del CSV de Kaggle:
        ```bash
        export DIABETES_CSV="data/diabetes.csv"
        ```
        """),
        code("""
        import os, numpy as np, pandas as pd
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
        output_dir = "outputs"; os.makedirs(output_dir, exist_ok=True)
        """),
        code("""
        print("🔹 Iniciando pipeline Clustering Basado en Densidad...")
        # Carga (devuelve ya escalado)
        X_scaled, y = cargar_dataset("diabetes", usar_minmax=True)
        X = X_scaled  # mantener referencia
        # PCA 2D
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        print(f"🔍 Varianza explicada por PCA: {pca.explained_variance_ratio_.sum():.2f}")
        # Visual base
        visualizar_clusters(X_pca, y, "PCA_2D_-_Clustering_Basado_en_Densidad")
        """),
        code("""
        # === DBSCAN (rejilla pequeña) ===
        resultados_dbscan = []
        for eps in [0.3, 0.5, 0.7]:
            for min_samples in [3, 5, 10]:
                print(f"📌 Evaluando DBSCAN eps={eps}, min_samples={min_samples}")
                etiquetas = aplicar_dbscan(X_pca, eps=eps, min_samples=min_samples)
                visualizar_clusters(X_pca, etiquetas, f"DBSCAN_eps_{eps}__min_samples_{min_samples}")
                resultados_dbscan.append(
                    evaluar_clusterings(X_pca, etiquetas, "DBSCAN", {"eps": eps, "min_samples": min_samples})
                )
        pd.DataFrame(resultados_dbscan).to_csv(f"{output_dir}/DBSCAN_consolidado.csv", index=False)
        print("✅ Resultados DBSCAN guardados en CSV consolidado")
        """),
        code("""
        # === UMAP (opcional) ===
        try:
            X_umap = aplicar_umap(X_scaled, n_components=2)
            visualizar_clusters(X_umap, y, "UMAP_Visualization")
        except Exception as e:
            print("UMAP no disponible o falló:", e)
        """),
        code("""
        # === HDBSCAN ===
        resultados_hdbscan = []
        for min_cluster_size in [5, 10]:
            print(f"📌 Evaluando HDBSCAN con min_cluster_size={min_cluster_size}")
            etiquetas = aplicar_hdbscan(X_scaled, min_cluster_size=min_cluster_size)
            # Nota: nuestra función de visual guarda png en outputs/
            visualizar_clusters(X_scaled.values, etiquetas, f"HDBSCAN_min_cluster_size_{min_cluster_size}")
            resultados_hdbscan.append(
                evaluar_clusterings(X_scaled, etiquetas, "HDBSCAN", {"min_cluster_size": min_cluster_size})
            )
        pd.DataFrame(resultados_hdbscan).to_csv(f"{output_dir}/HDBSCAN_consolidado.csv", index=False)
        print("✅ Resultados HDBSCAN guardados en CSV consolidado")
        """),
        code("""
        # === Anomalías ===
        anomalies_iso = detectar_anomalias_isolation_forest(X_scaled)
        anomalies_svm = detectar_anomalias_one_class_svm(X_scaled)
        df_out = X.copy()
        df_out['Anomaly Isolation Forest'] = anomalies_iso
        df_out['Anomaly One-Class SVM'] = anomalies_svm
        print("Pacientes considerados anómalos por Isolation Forest:")
        display(df_out[df_out['Anomaly Isolation Forest'] == -1].head(20))
        print("Pacientes considerados anómalos por One-Class SVM:")
        display(df_out[df_out['Anomaly One-Class SVM'] == -1].head(20))
        """),
        md("""
        ✅ Los gráficos se guardan en `outputs/` (por ejemplo:  
        `viz_PCA_2D_-_Clustering_Basado_en_Densidad.png`,  
        `viz_DBSCAN_eps_0.5__min_samples_10.png`,  
        `viz_HDBSCAN_min_cluster_size_5.png`,  
        `UMAP_Visualization.png`, etc.)
        """),
    ]
    return nb

def make_nb_02():
    nb = nbf.v4.new_notebook()
    nb.cells = []

    nb.cells += [
        md("""
        # 🔎 Comparación de Detección de Anomalías: Isolation Forest vs One-Class SVM
        Equivalente a `scripts/main_comparacion.py`.  
        Genera datos sintéticos, entrena ambos modelos, guarda **gráficos** y calcula **métricas**.
        """),
        md("""
        ## 🔧 Requisitos
        ```bash
        conda activate especialidadmachinelearning
        pip install seaborn
        ```
        """),
        code("""
        import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
        from src.utils import generar_datos, preprocesar_datos
        from src.modelos import entrenar_modelos
        from src.visualizador import visualizar_resultados
        from src.evaluador import evaluar
        sns.set(style='whitegrid')
        output_dir = "outputs"; os.makedirs(output_dir, exist_ok=True)
        """),
        code("""
        print("🔹 Iniciando comparación entre Isolation Forest y One-Class SVM...")
        # Datos + preprocesamiento
        X, y = generar_datos(n_samples=1000, n_features=10, weight_normales=0.95, seed=42)
        X_scaled, X_pca = preprocesar_datos(X)
        # Entrenamiento
        iso_pred, svm_pred, params = entrenar_modelos(X_scaled, if_contamination=0.05, svm_gamma=0.1, svm_nu=0.05)
        # Visual comparativa (se guarda automáticamente)
        _ = visualizar_resultados(X_pca, iso_pred, svm_pred, params)
        # Métricas y matrices de confusión (se guardan automáticamente)
        evaluar(y, iso_pred, "IsolationForest", params)
        evaluar(y, svm_pred, "OneClassSVM", params)
        """),
        md("""
        ✅ Salidas generadas en `outputs/`:  
        - `comp_anomalias_IF_vs_OCSVM_contamination...png`  
        - `cm_IsolationForest_contamination...png`  
        - `cm_OneClassSVM_contamination...png`
        """),
    ]
    return nb

def main():
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    nb1 = make_nb_01()
    nb2 = make_nb_02()

    nb1_path = out_dir / "01_clustering_pipeline_diabetes.ipynb"
    nb2_path = out_dir / "02_comparacion_anomalias.ipynb"

    with open(nb1_path, "w", encoding="utf-8") as f:
        nbf.write(nb1, f)
    with open(nb2_path, "w", encoding="utf-8") as f:
        nbf.write(nb2, f)

    print(f"✅ Notebook creado: {nb1_path}")
    print(f"✅ Notebook creado: {nb2_path}")

if __name__ == "__main__":
    main()
