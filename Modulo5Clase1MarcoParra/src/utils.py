import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def cargar_dataset():
    """
    Descarga y retorna el dataset 'adult' desde OpenML.

    Returns:
        pd.DataFrame: Dataset con información demográfica y laboral.
    """
    print("📥 Descargando dataset 'adult' desde OpenML...")
    data = fetch_openml(name="adult", version=2, as_frame=True)
    df = data.frame.copy()
    return df


def preprocesar_datos_clasificacion(df, test_size=0.2, random_state=42):
    """
    Realiza el preprocesamiento necesario para tareas de clasificación.

    Pasos:
        - Elimina filas con valores nulos.
        - Codifica la variable objetivo ('class') como etiquetas numéricas.
        - Aplica codificación one-hot a variables categóricas (dummies).
        - Estandariza las variables numéricas (Scaler).
        - Divide los datos en conjuntos de entrenamiento y prueba (estratificado).

    Args:
        df (pd.DataFrame): Dataset original.
        test_size (float): Proporción del conjunto de prueba.
        random_state (int): Semilla aleatoria para reproducibilidad.

    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    df = df.dropna()
    target_col = "class"

    # Codificar clase objetivo como números (LabelEncoder)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_col])

    # Eliminar la columna objetivo del dataset
    X = df.drop(columns=[target_col])

    # Codificar variables categóricas usando one-hot encoding
    # (dummies) para convertirlas en variables numéricas
    # y evitar problemas con modelos que no manejan texto directamente
    # drop_first=True para evitar la trampa de la variable ficticia
    # y reducir la multicolinealidad
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Escalar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)

    # Dividir en train/test conservando distribución de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, label_encoder


def preprocesar_datos_regresion(df, test_size=0.2, random_state=42):
    """
    Realiza el preprocesamiento necesario para tareas de regresión.

    Pasos:
        - Elimina filas con valores nulos.
        - Transforma la variable objetivo ('class') en valores numéricos float.
        - Codifica variables categóricas con one-hot encoding.
        - Estandariza las características numéricas.
        - Divide en train/test.

    Args:
        df (pd.DataFrame): Dataset original.
        test_size (float): Proporción del conjunto de prueba.
        random_state (int): Semilla aleatoria.

    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    df = df.dropna()
    target_col = "class"

    # Convertir clase objetivo a valores float
    y = df[target_col].map({'>50K': 1.0, '<=50K': 0.0}).astype(float)

    # Eliminar la columna objetivo
    X = df.drop(columns=[target_col])

    # Codificación one-hot
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Escalamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)

    # División simple en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler


def imprimir_resumen(X_train, X_test, y_train, y_test, caso):
    print(f"\n📊 Preprocesamiento para {caso.upper()}")
    print(f"🔹 X_train shape: {X_train.shape}")
    print(f"🔹 X_test shape: {X_test.shape}")
    print(f"🔹 y_train shape: {y_train.shape}")
    print(f"🔹 y_test shape: {y_test.shape}")
    print("\n🔍 Vista previa de X_train:")
    print(X_train.head())
