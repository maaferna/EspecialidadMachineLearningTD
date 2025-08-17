import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import re


def evaluar_clusterings(X, etiquetas, modelo, params, output_dir="outputs"):
    """
    Evalúa un clustering y devuelve resultados en forma de diccionario.

    Args:
        X (array-like): Datos de entrada.
        etiquetas (array-like): Etiquetas asignadas por el clustering.
        modelo (str): Nombre del modelo (DBSCAN o HDBSCAN).
        params (dict): Diccionario con los parámetros usados.
        output_dir (str): Carpeta donde guardar los resultados.

    Returns:
        dict: Resultados de métricas y parámetros.
    """
    # Filtrar clusters válidos (ignorar ruido con etiqueta -1)
    if len(set(etiquetas)) <= 1 or (set(etiquetas) == {-1}):
        silhouette = None
        davies = None
    else:
        silhouette = silhouette_score(X, etiquetas)
        davies = davies_bouldin_score(X, etiquetas)

    return {
        "modelo": modelo,
        **params,
        "n_clusters": len(set(etiquetas)) - (1 if -1 in etiquetas else 0),
        "silhouette": silhouette,
        "davies_bouldin": davies
    }




def aplicar_pca(X, n_componentes=2):
    """Aplica PCA y devuelve los componentes y la varianza explicada."""
    # Asegurar que los nombres de columnas sean strings
    if hasattr(X, 'columns'):
        X = X.copy()
        X.columns = X.columns.astype(str)
    
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X)
    varianza = pca.explained_variance_ratio_.cumsum()
    return X_pca, varianza, pca

def evaluar_knn_con_pca(X_train, y_train, X_test, y_test, n_componentes_list=[2,3], k_list=[3,5,7,9]):
    """Evalúa KNN con PCA como preprocesamiento."""
    # Asegurar que los nombres de columnas sean strings
    if hasattr(X_train, 'columns'):
        X_train = X_train.copy()
        X_train.columns = X_train.columns.astype(str)
    if hasattr(X_test, 'columns'):
        X_test = X_test.copy()
        X_test.columns = X_test.columns.astype(str)
    
    resultados = []
    for n_comp in n_componentes_list:
        pca = PCA(n_components=n_comp)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        for k in k_list:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)

            resultados.append({
                "n_componentes": n_comp,
                "n_neighbors": k,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred, average="macro")
            })

    return pd.DataFrame(resultados)




def pca_no_supervisado(X, n_componentes_list=[2, 3, 4]):
    """
    Aplica PCA en modo no supervisado y evalúa la varianza explicada acumulada.

    Args:
        X (pd.DataFrame): Dataset escalado.
        n_componentes_list (list): Lista de valores de componentes a evaluar.

    Returns:
        dict: Resultados con DataFrame y mejor número de componentes.
    """
    # 🔍 DEBUG INTENSIVO
    print(f"🔍 DEBUG - Entrada a pca_no_supervisado:")
    print(f"   - Tipo de X: {type(X)}")
    print(f"   - Shape de X: {X.shape}")
    print(f"   - Columnas: {list(X.columns) if hasattr(X, 'columns') else 'No tiene columnas'}")
    print(f"   - Primeros valores:\n{X.head(3) if hasattr(X, 'head') else X[:3] if len(X) > 0 else 'Dataset vacío'}")
    print(f"   - Tipos de datos: {X.dtypes if hasattr(X, 'dtypes') else 'No disponible'}")
    
    # 🔧 FIX: Asegurar que los nombres de columnas sean strings
    if hasattr(X, 'columns'):
        print(f"🔍 Tipos de nombres de columnas ANTES: {[type(col).__name__ for col in X.columns]}")
        X = X.copy()
        X.columns = X.columns.astype(str)
        print(f"🔍 Nombres de columnas DESPUÉS: {list(X.columns)}")
    
    # 🔧 FIX: Verificar NaN de forma más robusta
    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    print(f"🔍 Después de conversión a DataFrame:")
    print(f"   - Shape: {X_df.shape}")
    print(f"   - Tipos: {X_df.dtypes}")
    
    nan_count = X_df.isnull().sum().sum()
    
    if nan_count > 0:
        print(f"⚠️ {nan_count} NaN detectados en PCA")
        print(f"🔍 NaN por columna: {X_df.isnull().sum()}")
        print(f"🔍 Estadísticas antes de limpiar: {X_df.describe()}")
        
        X_clean = X_df.dropna()
        if len(X_clean) == 0:
            print("❌ DEBUG: Mostrando últimas filas antes de eliminar:")
            print(X_df.tail(10))
            raise ValueError("❌ Todos los datos fueron eliminados por NaN. Revisar el preprocessing.")
        X = X_clean
        print(f"✅ Dataset limpio: {X.shape[0]} filas, {X.shape[1]} features")
    else:
        print(f"✅ No se detectaron NaN: {X_df.shape[0]} filas, {X_df.shape[1]} features")
        X = X_df
    
    # Verificación adicional antes de PCA
    if len(X) == 0:
        raise ValueError("❌ Dataset vacío después del preprocessing")
    
    print(f"🔍 Datos finales para PCA: Shape={X.shape}, NaN={X.isnull().sum().sum()}")
    
    resultados = []

    for n in n_componentes_list:
        if n > X.shape[1]:
            print(f"⚠️ Saltando n_componentes={n} (mayor que {X.shape[1]} features)")
            continue
        if n > len(X):
            print(f"⚠️ Saltando n_componentes={n} (mayor que {len(X)} muestras)")
            continue
            
        pca = PCA(n_components=n)
        pca.fit(X)
        varianza_acum = np.sum(pca.explained_variance_ratio_)
        resultados.append({"n_componentes": n, "varianza_explicada": varianza_acum})

    if not resultados:
        raise ValueError("❌ No se pudieron calcular componentes PCA válidos")

    df_resultados = pd.DataFrame(resultados)

    df_resultados.to_csv("outputs/pca_resultados.csv", index=False)
    
    # Seleccionar el mejor número de componentes (mayor o igual a 0.95)
    mejor = df_resultados[df_resultados["varianza_explicada"] >= 0.95]

    if mejor.empty:
        print("⚠️ No se encontró un número de componentes que explique al menos el 95% de la varianza.")
        mejor_n = None
    else:
        mejor_n = mejor.iloc[0]["n_componentes"]  # Tomar el primer valor que cumpla la condición

    return {
        "mejor_n": int(mejor_n) if mejor_n is not None else None,
        "resultados": df_resultados
    }

def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(name))


def evaluar(y_true, y_pred, modelo, params=None):
    # ===== Config =====
    sns.set(style='whitegrid')
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    """Evalúa el rendimiento del modelo y guarda la matriz de confusión."""
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Guardar matriz de confusión como imagen
    fname = f"cm_{modelo}"
    if params:
        fname += "_" + "_".join(f"{k}{_sanitize(v)}" for k, v in params.items())
    cm_path = os.path.join(OUTPUT_DIR, f"{_sanitize(fname)}.png")

    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format="d", cmap="Blues")
    plt.title(f"Matriz de Confusión - {modelo}")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    print(f"Modelo: {modelo}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC AUC: {roc:.2f}")
    print("Matriz de confusión:")
    print(cm)
    print(f"🖼️ Matriz de confusión guardada en: {cm_path}\n")
    return {"f1": f1, "roc_auc": roc, "cm_path": cm_path}