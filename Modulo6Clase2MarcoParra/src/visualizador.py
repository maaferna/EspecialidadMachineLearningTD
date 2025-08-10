import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def graficar_clusters(X, etiquetas, title, filename):
    """Grafica clusters en 2D usando las primeras dos columnas."""
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], hue=etiquetas, palette="viridis", legend="full")
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"📊 Gráfico guardado en {filename}")

def graficar_pca(X, etiquetas, title, filename):
    """Aplica PCA a 2 componentes y grafica clusters."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=etiquetas, palette="viridis", legend="full")
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"📊 PCA guardado en {filename}")

def graficar_tsne(X, etiquetas, title, filename):
    """Aplica t-SNE y grafica resultados."""
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=etiquetas, palette="viridis", legend="full")
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"📊 t-SNE guardado en {filename}")
