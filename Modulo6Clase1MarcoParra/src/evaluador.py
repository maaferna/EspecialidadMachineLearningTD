import pandas as pd

def guardar_etiquetas(X, etiquetas, metodo, clusters, output_path):
    """Guarda las etiquetas de los clusters en un archivo CSV."""
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["cluster"] = etiquetas
    df.to_csv(f"{output_path}/{metodo}_{clusters}_clusters.csv", index=False)
