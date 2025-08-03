import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split



def aplicar_pca(X, n_componentes=2):
    """
    Aplica PCA a un dataset estandarizado.
    
    Args:
        X (pd.DataFrame): Dataset estandarizado.
        n_componentes (int): Número de componentes principales.

    Returns:
        X_pca (pd.DataFrame): Dataset proyectado.
        pca (PCA): Objeto PCA entrenado.
    """
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X)
    return X_pca, pca



def evaluar_pca(X, n_componentes_list=[2, 3, 4, 5, 6, 7, 8, 9], umbral_varianza=0.95, output_dir="outputs"):
    """
    Evalúa PCA para distintos n_componentes y selecciona el óptimo según varianza acumulada.

    Args:
        X (pd.DataFrame): Dataset completo (sin división train/test).
        n_componentes_list (list): Lista de valores de n_components a evaluar.
        umbral_varianza (float): Umbral mínimo de varianza acumulada para considerar óptimo.
        output_dir (str): Directorio donde guardar resultados.

    Returns:
        int: Número de componentes óptimo según el umbral.
    """
    print("🔹 Evaluando PCA exploratorio...")

    resultados = []
    for n in n_componentes_list:
        pca = PCA(n_components=n)
        pca.fit(X)
        varianza_acum = pca.explained_variance_ratio_.sum()
        resultados.append({"n_componentes": n, "varianza_explicada": varianza_acum})
        print(f"  - {n} componentes → varianza acumulada: {varianza_acum:.4f}")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(f"{output_dir}/pca_resultados.csv", index=False)

    # Seleccionar óptimo
    df_filtrado = df_resultados[df_resultados["varianza_explicada"] >= umbral_varianza]
    if not df_filtrado.empty:
        n_optimo = int(df_filtrado.iloc[0]["n_componentes"])
    else:
        n_optimo = int(df_resultados.iloc[-1]["n_componentes"])  # fallback al mayor

    print(f"✅ PCA óptimo seleccionado: {n_optimo} componentes (≥ {umbral_varianza*100:.0f}% varianza)")
    return n_optimo



def evaluar_knn(X, y, usar_pca=False, n_componentes=2, n_neighbors=5):
    """
    Entrena y evalúa un modelo KNN con o sin PCA como preprocesamiento.
    
    Args:
        X (pd.DataFrame): Dataset estandarizado.
        y (pd.Series): Etiquetas del dataset.
        usar_pca (bool): Si True, aplica PCA antes de KNN.
        n_componentes (int): Número de componentes principales para PCA.
        n_neighbors (int): Número de vecinos en KNN.
    
    Returns:
        dict: Métricas del modelo entrenado.
    """
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if usar_pca:
        pipeline = Pipeline([
            ("pca", PCA(n_components=n_componentes)),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors))
        ])
    else:
        pipeline = Pipeline([
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors))
        ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "usar_pca": usar_pca,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="macro")
    }


def evaluar_knn_con_pca_grid(X, y, componentes=[2, 3, 5], vecinos=[3, 5, 7]):
    """
    Evalúa KNN con distintas combinaciones de número de componentes PCA y vecinos KNN.

    Args:
        X (pd.DataFrame): Dataset estandarizado.
        y (pd.Series): Etiquetas.
        componentes (list): Lista de números de componentes PCA a probar.
        vecinos (list): Lista de valores de n_neighbors para KNN.

    Returns:
        pd.DataFrame: Resultados de todas las combinaciones.
    """
    resultados = []
    for n_comp in componentes:
        for k in vecinos:
            res = evaluar_knn(X, y, usar_pca=True, n_componentes=n_comp, n_neighbors=k)
            res["n_componentes"] = n_comp
            res["n_neighbors"] = k
            resultados.append(res)
    
    return pd.DataFrame(resultados)

