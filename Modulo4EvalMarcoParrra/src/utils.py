import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def cargar_datos(path="data/Training.csv"):
    """
    Carga el dataset de enfermedades múltiples desde archivo CSV.
    """

    df = pd.read_csv(path)
    print(f"✅ Dataset cargado con forma: {df.shape}")
    print("🔍 Primeras filas:")
    print(df.head())
    # Eliminar columnas tipo "Unnamed"
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    print(f"✅ Dataset cargado con forma: {df.shape}")
    return df


def preprocesar_datos(df, aplicar_escalado=False):
    """
    Preprocesamiento recomendado para clasificación multiclase:
    - Elimina columnas constantes
    - Elimina columnas con frecuencia de activación < 1%
    - Imputa valores
    - Codifica etiquetas multiclase
    - (Opcional) Escala variables si se requiere
    - Divide en train/test de forma estratificada
    """

    # Separar X y y
    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]

    # Eliminar columnas constantes
    X = X.loc[:, X.nunique() > 1]

    # Eliminar columnas con baja frecuencia de activación
    freq_1 = (X.sum(axis=0) / len(X))
    cols_to_keep = freq_1[freq_1 > 0.01].index
    X = X[cols_to_keep]

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Imputación
    imputer = SimpleImputer(strategy="most_frequent")
    X_imputed = imputer.fit_transform(X)

    # Escalado opcional
    if aplicar_escalado:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_imputed)
    else:
        X_processed = X_imputed

    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, label_encoder

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


def preprocesar_datos_multiclase_cv(df, fit_scaler=False, scaler=None, label_encoder=None):
    """
    Preprocesamiento adaptable a entrenamiento y testeo:
    - Elimina columnas constantes
    - Elimina columnas con frecuencia de activación < 1%
    - Imputa valores
    - Escala variables si se requiere
    - Codifica etiquetas con LabelEncoder compartido
    """

    # Separar X e y
    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]

    # Eliminar columnas constantes
    X = X.loc[:, X.nunique() > 1]

    # Eliminar columnas con baja frecuencia de activación
    freq_1 = (X.sum(axis=0) / len(X))
    cols_to_keep = freq_1[freq_1 > 0.01].index
    X = X[cols_to_keep]

    # Imputación
    imputer = SimpleImputer(strategy="most_frequent")
    X_imputed = imputer.fit_transform(X)

    # Escalado
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        X_scaled = scaler.transform(X_imputed)

    # Codificación de etiquetas
    if fit_scaler:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = label_encoder.transform(y)

    return X_scaled, y_encoded, scaler, label_encoder
