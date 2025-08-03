import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



def graficar_varianza_pca(output_dir="outputs", path_to=""):
    """Genera gráfico de varianza explicada acumulada desde CSV."""
    
    df = pd.read_csv(path_to)
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="n_componentes", y="varianza_explicada", marker="o")
    plt.axhline(0.95, color="red", linestyle="--", label="Umbral 95%")
    plt.title("Varianza Explicada Acumulada - PCA")
    plt.xlabel("Número de Componentes")
    plt.ylabel("Varianza Explicada Acumulada")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_varianza_explicada.png"))
    plt.close()
    print(f"📊 Gráfico de varianza PCA guardado en {output_dir}/pca_varianza_explicada.png")


def graficar_pca_2d(X_pca, y=None, output_path="outputs/pca_proyeccion_2d.png"):
    """Grafica la proyección 2D del PCA. Si hay etiquetas, colorea por clase."""
    plt.figure(figsize=(8, 6))
    if y is not None:
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="Set2", s=60)
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=60)
    plt.title("Proyección 2D con PCA")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Proyección PCA 2D guardada en {output_path}")


def graficar_heatmap_knn(resultados, output_dir="outputs"):
    """Heatmap de accuracy de KNN con PCA."""
    pivot = resultados.pivot(index="n_componentes", columns="n_neighbors", values="accuracy")
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Accuracy KNN con PCA")
    plt.xlabel("Vecinos (K)")
    plt.ylabel("Componentes PCA")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_knn_pca.png")
    plt.close()



def graficar_pca_3d(X_pca, y=None, title="Proyección PCA 3D", output_dir="outputs", nombre_archivo="pca_cluster_3d.png"):
    """
    Genera un gráfico 3D de los datos proyectados por PCA.

    Args:
        X_pca (pd.DataFrame o np.ndarray): Datos transformados por PCA (3 componentes).
        y (array-like, opcional): Etiquetas de clase o clusters para colorear puntos.
        title (str): Título del gráfico.
        output_dir (str): Carpeta donde guardar la imagen.
        nombre_archivo (str): Nombre del archivo de salida.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(X_pca, pd.DataFrame):
        X_plot = X_pca.values
    else:
        X_plot = X_pca

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if y is not None:
        scatter = ax.scatter(
            X_plot[:, 0], X_plot[:, 1], X_plot[:, 2],
            c=y, cmap="viridis", s=50, alpha=0.8
        )
        legend1 = ax.legend(*scatter.legend_elements(), title="Clases")
        ax.add_artist(legend1)
    else:
        ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], c="blue", s=50, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.set_zlabel("Componente 3")

    plt.tight_layout()
    path = os.path.join(output_dir, nombre_archivo)
    plt.savefig(path)
    plt.close()
    print(f"📊 Gráfico PCA 3D guardado en {path}")



def graficar_pca_2d(X_pca, y, title="PCA 2D", output_dir="outputs", nombre_archivo="pca_2d.png"):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="Set1", s=60, edgecolor="k")
    plt.title(title)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{nombre_archivo}")
    plt.close()
    print(f"📊 Gráfico PCA 2D guardado en {output_dir}/{nombre_archivo}")

