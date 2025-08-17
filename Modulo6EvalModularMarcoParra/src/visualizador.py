import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import re

from src.evaluador import _sanitize

# Carpeta de salida por defecto
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _sanitize_filename(name: str) -> str:
    """Convierte texto a nombre de archivo seguro."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)

def graficar_clusters(X, etiquetas, title, params=None):
    """
    Grafica clusters en 2D usando las primeras dos columnas del DataFrame/array.
    Guarda en outputs con nombre descriptivo.
    """
    if hasattr(X, "iloc"):
        x0, x1 = X.iloc[:, 0], X.iloc[:, 1]
    else:
        X = np.asarray(X)
        x0, x1 = X[:, 0], X[:, 1]

    filename = _sanitize_filename(f"clusters_{title}")
    if params:
        filename += "_" + "_".join(f"{k}{v}" for k, v in params.items())
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x0, y=x1, hue=etiquetas, palette="viridis", legend="full")
    plt.title(title)
    plt.savefig(filepath, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"📊 Gráfico guardado en {filepath}")

def graficar_pca(X, etiquetas, title, params=None):
    """Aplica PCA a 2 componentes y grafica clusters."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    filename = _sanitize_filename(f"pca_{title}")
    if params:
        filename += "_" + "_".join(f"{k}{v}" for k, v in params.items())
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=etiquetas, palette="viridis", legend="full")
    plt.title(title)
    plt.savefig(filepath, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"📊 PCA guardado en {filepath}")

def graficar_tsne(X, etiquetas, title, params=None):
    """Aplica t-SNE y grafica resultados."""
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    filename = _sanitize_filename(f"tsne_{title}")
    if params:
        filename += "_" + "_".join(f"{k}{v}" for k, v in params.items())
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=etiquetas, palette="viridis", legend="full")
    plt.title(title)
    plt.savefig(filepath, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"📊 t-SNE guardado en {filepath}")

def visualizar_clusters(X, labels, title, params=None):
    """
    Visualiza clusters en un gráfico 2D (ndarray o DataFrame).
    Guarda en outputs con nombre descriptivo.
    """
    if hasattr(X, "values"):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)

    filename = _sanitize_filename(f"viz_{title}")
    if params:
        filename += "_" + "_".join(f"{k}{v}" for k, v in params.items())
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.png")

    plt.figure(figsize=(10, 6))
    plt.scatter(X_arr[:, 0], X_arr[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title(title)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.colorbar()
    plt.savefig(filepath, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"📊 Visualización guardada en {filepath}")




def visualizar_resultados(X_pca, iso_pred, svm_pred, params, prefix="comp_anomalias"):
    """Visualiza y guarda los resultados de detección de anomalías."""
    # Figura comparativa IF vs OCSVM
    fname = f"{prefix}_IF_vs_OCSVM" + "_" + "_".join(f"{k}{_sanitize(v)}" for k, v in params.items())
    filepath = os.path.join(OUTPUT_DIR, f"{_sanitize(fname)}.png")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].scatter(X_pca[iso_pred == 0, 0], X_pca[iso_pred == 0, 1], label="Normal", alpha=0.6)
    axs[0].scatter(X_pca[iso_pred == 1, 0], X_pca[iso_pred == 1, 1], label="Anómalo", c='r', edgecolors='k')
    axs[0].set_title("Isolation Forest"); axs[0].legend()

    axs[1].scatter(X_pca[svm_pred == 0, 0], X_pca[svm_pred == 0, 1], label="Normal", alpha=0.6)
    axs[1].scatter(X_pca[svm_pred == 1, 0], X_pca[svm_pred == 1, 1], label="Anómalo", c='r', edgecolors='k')
    axs[1].set_title("One-Class SVM"); axs[1].legend()

    plt.suptitle("Comparación visual de detección de anomalías", fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"📊 Gráfico comparativo guardado en: {filepath}")
    return filepath
