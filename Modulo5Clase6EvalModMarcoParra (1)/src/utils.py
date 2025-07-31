import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

def cargar_y_preprocesar_credit(test_size: float = 0.2, random_state: int = 42):
    """
    Carga y preprocesa el dataset 'credit' desde OpenML.
    Aplica limpieza, imputación, codificación de categóricas y escalamiento.

    Args:
        test_size (float): Proporción para conjunto de prueba.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        X_train (pd.DataFrame): Features de entrenamiento.
        X_test (pd.DataFrame): Features de prueba.
        y_train (pd.Series): Target de entrenamiento.
        y_test (pd.Series): Target de prueba.

    Carga y preprocesa el dataset 'credit' desde OpenML.
    Incluye imputación de nulos, escalado de numéricas y codificación
    de categóricas si existen.
    """
    print("🔹 Iniciando pipeline para el dataset 'credit'")
    print("📥 Cargando dataset 'credit' desde OpenML...")

    dataset = fetch_openml("credit", version=1, as_frame=True)
    df = dataset.frame.copy()
    target_col = "SeriousDlqin2yrs"

    print(f"✅ Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
    print(f"🎯 Columna target detectada: '{target_col}'")

    # Manejo de nulos
    if df.isnull().sum().sum() > 0:
        print(f"⚠️ Se encontraron {df.isnull().sum().sum()} valores nulos. Imputando...")
    else:
        print("✅ No se encontraron valores nulos explícitos.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Detectar columnas categóricas y numéricas
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"🔍 Columnas categóricas: {len(cat_cols)} → {cat_cols}")
    print(f"🔍 Columnas numéricas: {len(num_cols)} → {num_cols}")

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers = [("num", num_pipeline, num_cols)]

    if cat_cols:  # Solo si existen categóricas
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers)

    # Aplicar transformaciones
    X_processed = preprocessor.fit_transform(X)

    # Recuperar nombres de columnas
    feature_names = num_cols
    try:
        if cat_cols:
            feature_names += list(preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(cat_cols))
    except Exception:
        print("⚠️ No se generaron variables categóricas (no existían).")

    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    # División train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed_df, y, test_size=0.2, random_state=42
    )
    print(f"✅ División realizada: X_train={X_train.shape}, X_test={X_test.shape}")

    return X_train, X_test, y_train, y_test
