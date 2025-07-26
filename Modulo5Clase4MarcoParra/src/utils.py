# src/pipeline.py

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def cargar_dataset():
    """
    Carga el dataset desde OpenML y retorna un DataFrame.
    """
    print("📥 Descargando dataset 'adult' desde OpenML...")
    dataset = fetch_openml("adult", version=2, as_frame=True)
    df = dataset.frame
    print("✅ Dataset cargado exitosamente.")
    print(f"📊 Tamaño del dataset: {df.shape[0]} filas y {df.shape[1]} columnas.")
    print("🔍 Primeras filas del dataset:")
    print(df.head())
    return df

def preprocesar_datos(df):
    """
    Preprocesa el dataset para clasificación binaria de abandono de servicio.

    Pasos:
        - Elimina filas con valores nulos.
        - Transforma la variable objetivo ('class') en valores binarios (1.0 / 0.0).
        - Codifica variables categóricas con one-hot encoding.
        - Estandariza las características numéricas.

    Args:
        df (pd.DataFrame): Dataset original.

    Returns:
        tuple: X (features), y (target), scaler (ajustado)
    """
    df = df.dropna()
    target_col = "class"

    # Convertir clase objetivo a valores binarios float
    y = df[target_col].map({">50K": 1.0, "<=50K": 0.0}).astype(float)

    # Eliminar la columna objetivo
    X = df.drop(columns=[target_col])

    # Codificación one-hot
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Escalamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)

    return X_scaled, y, scaler
