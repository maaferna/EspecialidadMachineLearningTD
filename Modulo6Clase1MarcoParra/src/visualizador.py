import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def graficar_dendrograma(Z, titulo, output_path):
    """Genera un dendrograma a partir de la matriz de enlace Z.
    Z: Matriz de enlace generada por scipy.cluster.hierarchy.linkage
    titulo: Título del gráfico
    output_path: Ruta donde guardar la imagen del dendrograma
    """
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.title(titulo)
    plt.savefig(output_path)
    plt.close()


def graficar_tsne(X, etiquetas, titulo, output_path):
    """
    Genera un gráfico t-SNE 2D de los datos.
    X: Datos a graficar (debe ser un array 2D)
    etiquetas: Etiquetas de los clusters o clases para colorear los puntos
    titulo: Título del gráfico
    output_path: Ruta donde guardar la imagen del gráfico
    
    """
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=etiquetas, palette="Set2")
    plt.title(titulo)
    plt.savefig(output_path)
    plt.close()




def graficar_pca_2d(X_pca, y=None, title="PCA 2D", output_dir="outputs"):
    """
    Genera un gráfico 2D de los datos proyectados con PCA.
    
    Args:
        X_pca (np.ndarray): Datos transformados por PCA (2 componentes).
        y (array-like): Etiquetas verdaderas (opcional).
        title (str): Título del gráfico.
    """
    plt.figure(figsize=(8, 6))
    if y is not None:
        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            hue=y,
            palette="viridis",
            alpha=0.8,
            edgecolor="k"
        )
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8, edgecolor="k")

    plt.title(title)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_2d.png")
    plt.close()


def graficar_pca_3d(X_pca, y=None, title="PCA 3D", output_dir="outputs"):
    """
    Genera un gráfico 3D de los datos proyectados con PCA.

    Args:
        X_pca (np.ndarray): Datos transformados por PCA (3 componentes).
        y (array-like): Etiquetas verdaderas (opcional).
        title (str): Título del gráfico.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if y is not None:
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
            c=y, cmap="viridis", alpha=0.8, edgecolor="k"
        )
        legend1 = ax.legend(*scatter.legend_elements(), title="Clases")
        ax.add_artist(legend1)
    else:
        ax.scatter(
            X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
            alpha=0.8, edgecolor="k"
        )

    ax.set_title(title)
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.set_zlabel("Componente Principal 3")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_3d.png")
    plt.close()

