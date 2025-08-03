from sklearn.decomposition import PCA
from src.utils import cargar_dataset
from src.modelos import clustering_jerarquico
from src.visualizador import graficar_dendrograma, graficar_pca_2d, graficar_pca_3d, graficar_tsne
from src.evaluador import guardar_etiquetas
import os

def main():
    print("🔹 Iniciando pipeline Clustering Jerárquico...")

    X, y = cargar_dataset("iris")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)


    # PCA 2D
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)
    graficar_pca_2d(X_pca_2d, y, title="PCA 2D - Clustering Jerárquico")

    # PCA 3D
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X)
    graficar_pca_3d(X_pca_3d, y, title="PCA 3D - Clustering Jerárquico")

    for metodo in ["ward", "average"]:
        for n_clusters in [2, 3]:
            print(f"📌 {metodo.upper()} con {n_clusters} clusters")

            Z, etiquetas = clustering_jerarquico(X, metodo=metodo, n_clusters=n_clusters)

            graficar_dendrograma(Z, f"Dendrograma - {metodo} ({n_clusters} clusters)",
                                 f"{output_dir}/dendrograma_{metodo}_{n_clusters}.png")


            graficar_tsne(X, etiquetas, f"t-SNE - {metodo} ({n_clusters} clusters)",
                          f"{output_dir}/tsne_{metodo}_{n_clusters}.png")

            guardar_etiquetas(X, etiquetas, metodo, n_clusters, output_dir)

if __name__ == "__main__":
    main()
