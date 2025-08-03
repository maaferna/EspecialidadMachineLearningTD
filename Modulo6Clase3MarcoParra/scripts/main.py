# scripts/main.py
import os
import pandas as pd
from sklearn.decomposition import PCA
from src.utils import cargar_dataset
from src.evaluador import pca_no_supervisado, evaluar_knn_con_pca
from src.visualizador import graficar_pca_2d, graficar_pca_3d, graficar_varianza_pca, graficar_heatmap_knn

def main():
    print("🔹 Iniciando pipeline PCA -> KNN...")

    # --- 1. Cargar dataset ---
    nombre_dataset = "iris"   # o "wine"
    usar_minmax = True
    X, y = cargar_dataset(
        nombre_dataset=nombre_dataset, 
        usar_minmax=usar_minmax, 
        con_split=False
    )

    # --- 2. Evaluar PCA no supervisado ---
    print("📊 Evaluando PCA no supervisado...")
    resultado_pca = pca_no_supervisado(
        X,  # ← CORREGIDO: Solo pasar X, no concatenar con y
        n_componentes_list=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    n_componentes_optimo = resultado_pca["mejor_n"] # De acuerdo a la gràfica el parametro optimo es 2, porque se logra un 95% de explicación de varianza
    print(f"✅ PCA óptimo: {n_componentes_optimo} componentes")
    graficar_varianza_pca("outputs","outputs/pca_resultados.csv")
    print("✅ Resultados PCA:")
    print("Numero optimo PCA:", n_componentes_optimo)
    print("Resultados detallados:")
    print(resultado_pca["resultados"])

    # Aplicar PCA a 2 componentes, òptimo según el análisis
    pca_2d = PCA(n_components=n_componentes_optimo)
    X_pca_2d = pca_2d.fit_transform(X)  # O usa X_train si quieres solo entrenamiento

    # Graficar con clases verdaderas (y)
    graficar_pca_2d(X_pca_2d, y, title="PCA en 2D con clases reales, òptimo 2 componentes")

    # Aplicar PCA a 3 componentes
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X)  # O usa X_train si quieres solo entrenamiento

    # Graficar con clases verdaderas (y)
    graficar_pca_3d(X_pca_3d, y, title="PCA en 3D con clases reales")

    # (Opcional) Graficar con clusters si usas KMeans
    from sklearn.cluster import KMeans
    clusters = KMeans(n_clusters=3, random_state=42).fit_predict(X_pca_3d)
    graficar_pca_3d(X_pca_3d, clusters, title="PCA en 3D con KMeans")


    
    # Graficar varianza PCA (si la función existe)
    try:
        graficar_varianza_pca(resultado_pca["resultados"])
    except Exception as e:
        print(f"⚠️ No se pudo graficar varianza: {e}")

    # --- 3. Fase 2: PCA + KNN supervisado ---
    print("📊 Evaluando KNN con PCA...")
    X_train, X_test, y_train, y_test = cargar_dataset(
        nombre_dataset, usar_minmax=usar_minmax, con_split=True
    )
    
    resultados_knn = evaluar_knn_con_pca(
        X_train, y_train, X_test, y_test, 
        n_componentes_list=[n_componentes_optimo], 
        k_list=[3, 5, 7, 9]
    )

    print("✅ Resultados KNN con PCA óptimo:")
    print(resultados_knn)

    # Graficar heatmap de accuracies (si la función existe)
    try:
        graficar_heatmap_knn(resultados_knn)
    except Exception as e:
        print(f"⚠️ No se pudo graficar heatmap: {e}")

if __name__ == "__main__":
    main()