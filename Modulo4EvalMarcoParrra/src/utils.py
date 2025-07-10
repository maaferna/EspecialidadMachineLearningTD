import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def cargar_datos_breast_cancer():
    """Carga el dataset Breast Cancer desde sklearn y lo convierte a un DataFrame."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    print("\n🧬 Dataset Breast Cancer cargado:")
    print(df.head())
    print("\n📊 Información general:")
    print(df.info())
    print("\n📈 Estadísticas:")
    print(df.describe())

    return df


def preprocesar_datos(df):
    """Escala los datos y separa en conjuntos de entrenamiento y prueba (70/30)."""
    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
