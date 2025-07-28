# src/utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle

def cargar_y_preprocesar_adult_income(ruta: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"):
    """
    Carga y preprocesa el dataset Adult Income del UCI Repository.
    Aplica limpieza, codificación y escalamiento siguiendo buenas prácticas.

    Args:
    ruta (str): Ruta al archivo .csv del dataset.

    Returns:
    X_train (pd.DataFrame): Conjunto de entrenamiento con features.
    X_test (pd.DataFrame): Conjunto de prueba con features.
    y_train (pd.Series): Target de entrenamiento.
    y_test (pd.Series): Target de prueba.
    """
    print("📥 Cargando dataset Adult Income...")

    columnas = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]

    df = pd.read_csv(ruta, header=None, names=columnas, na_values=" ?")
    print(f"✅ Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")

    # Eliminar filas con valores nulos
    nulos = df.isnull().sum().sum()
    df.dropna(inplace=True)
    print(f"🧹 Filas con nulos eliminadas: {nulos}")

    # Codificar target binario
    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)
    print("🎯 Columna target binarizada (income > 50K = 1)")

    # Separar features y target
    X = df.drop("income", axis=1)
    y = df["income"]

    # Detectar columnas numéricas y categóricas
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"🔍 Columnas categóricas: {len(cat_cols)} → {cat_cols}")
    print(f"🔍 Columnas numéricas: {len(num_cols)} → {num_cols}")

    # Pipeline para numéricas: imputación + escalamiento
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])

    # Pipeline para categóricas: imputación + one-hot encoding
    cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # ColumnTransformer para aplicar ambos pipelines
    preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
    ])

    # Aplicar transformaciones
    X_processed = preprocessor.fit_transform(X)
    feature_names = (
    num_cols + list(preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(cat_cols))
    )
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    # Mezclar aleatoriamente para evitar orden
    X_processed_df, y = shuffle(X_processed_df, y, random_state=42)

    # Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42)
    print(f"✅ División realizada: X_train={X_train.shape}, X_test={X_test.shape}")

    return X_train, X_test, y_train, y_test