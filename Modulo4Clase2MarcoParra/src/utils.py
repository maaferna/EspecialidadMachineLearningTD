import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import time

# URL del dataset sin encabezados
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def cargar_dataset():
    """
    Carga el dataset desde la URL oficial y asigna nombres a las columnas.
    """
    df = pd.read_csv(URL, header=None, names=COLUMN_NAMES)
    print("\n✨ Primeras filas del dataset:")
    print(df.head())

    print("\nℹ️ Información general:")
    print(df.info())

    print("\n📈 Estadísticas descriptivas:")
    print(df.describe())

    return df


def preprocesar_datos(df):
    """
    Aplica escalamiento a las variables predictoras y separa conjuntos de entrenamiento y prueba.
    """
    print("\n🔄 Normalizando datos...")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def entrenar_modelo_base(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo base con RandomForest sin optimización de hiperparámetros.
    """
    print("\n🌲 Entrenando modelo base...")

    start = time.time()
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n📊 Evaluación del modelo base:")
    print(classification_report(y_test, y_pred))
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Tiempo de entrenamiento: {end - start:.2f} segundos")

    return {
        "metodo": "Base",
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "modelo": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "tiempo": end - start,
        "mejores_parametros": None,
    }
