# scripts/crear_notebook.py

import nbformat as nbf
import os

def crear_notebook():
    # Asegurar directorio notebooks
    notebook_dir = "notebooks"
    os.makedirs(notebook_dir, exist_ok=True)

    notebook_path = os.path.join(notebook_dir, "credit_scoring_notebook.ipynb")

    # Crear notebook
    nb = nbf.v4.new_notebook()

    # 1️⃣ Celda: pip install
    nb['cells'].append(nbf.v4.new_code_cell(
        """!pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost shap lime"""
    ))

    # 2️⃣ Celda: Imports y configuración
    nb['cells'].append(nbf.v4.new_code_cell(
        """import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# Inline plots
%matplotlib inline

# Configuración general
plt.rcParams['figure.figsize'] = (10,6)
plt.style.use('seaborn-v0_8')

print("✅ Librerías importadas correctamente.")"""
    ))

    # 3️⃣ Celda: Cargar y preprocesar datos
    nb['cells'].append(nbf.v4.new_code_cell(
        """from src.utils import cargar_y_preprocesar_credit

X_train, X_test, y_train, y_test = cargar_y_preprocesar_credit()
print(f"✅ Datos listos: X_train={X_train.shape}, X_test={X_test.shape}")"""
    ))

    # 4️⃣ Celda: Entrenar modelos
    nb['cells'].append(nbf.v4.new_code_cell(
        """from src.modelos import entrenar_logistic_regression, entrenar_random_forest

resultados_logistic = entrenar_logistic_regression(X_train, y_train, X_test, y_test)
resultados_rf = entrenar_random_forest(X_train, y_train, X_test, y_test)"""
    ))

    # 5️⃣ Celda: Visualización de resultados
    nb['cells'].append(nbf.v4.new_code_cell(
        """from src.visualizador import (
    graficar_matriz_confusion,
    graficar_curva_roc,
    graficar_importancia_variables
)

# Logistic Regression
graficar_matriz_confusion(resultados_logistic["resultados"]["confusion_matrix"], "LogisticRegression", output_dir="../outputs")
graficar_curva_roc(resultados_logistic["modelo"], X_test, y_test, "LogisticRegression", output_dir="../outputs")
graficar_importancia_variables(resultados_logistic["modelo"], X_train, "LogisticRegression", output_dir="../outputs")

# Random Forest
graficar_matriz_confusion(resultados_rf["resultados"]["confusion_matrix"], "RandomForest", output_dir="../outputs")
graficar_curva_roc(resultados_rf["modelo"], X_test, y_test, "RandomForest", output_dir="../outputs")
graficar_importancia_variables(resultados_rf["modelo"], X_train, "RandomForest", output_dir="../outputs")

print("✅ Visualizaciones generadas correctamente.")"""
    ))

    # 6️⃣ Celda: Mostrar imágenes generadas
    nb['cells'].append(nbf.v4.new_code_cell(
        """imagenes = [
    "../outputs/confusion_matrix_logisticregression.png",
    "../outputs/roc_curve_logisticregression.png",
    "../outputs/feature_importance_logisticregression.png",
    "../outputs/confusion_matrix_randomforest.png",
    "../outputs/roc_curve_randomforest.png",
    "../outputs/feature_importance_randomforest.png",
]

for img in imagenes:
    if os.path.exists(img):
        display(Image(filename=img))
    else:
        print(f"⚠️ Imagen no encontrada: {img}")"""
    ))

    # Guardar notebook
    with open(notebook_path, "w") as f:
        nbf.write(nb, f)

    print(f"✅ Notebook creado en: {notebook_path}")


if __name__ == "__main__":
    crear_notebook()
