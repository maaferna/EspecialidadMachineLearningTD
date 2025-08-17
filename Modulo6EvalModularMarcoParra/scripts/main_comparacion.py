import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from src.visualizador import visualizar_resultados
from src.utils import preprocesar_datos, generar_datos
from src.evaluador import evaluar
from src.modelos import entrenar_modelos






def main():
    print("🔹 Iniciando comparación entre Isolation Forest y One-Class SVM...")

    # 1) Datos + preprocesamiento
    X, y = generar_datos(n_samples=1000, n_features=10, weight_normales=0.95, seed=42)
    X_scaled, X_pca = preprocesar_datos(X)

    # 2) Entrenamiento (usar X_scaled!)
    iso_pred, svm_pred, params = entrenar_modelos(X_scaled, if_contamination=0.05, svm_gamma=0.1, svm_nu=0.05)

    # 3) Visualización comparativa (guardada)
    _ = visualizar_resultados(X_pca, iso_pred, svm_pred, params)

    # 4) Evaluación (y guardado de matrices)
    evaluar(y, iso_pred, "IsolationForest", params)
    evaluar(y, svm_pred, "OneClassSVM", params)

    # 5) Reflexión
    print("\nReflexión práctica:")
    print("• Isolation Forest suele ser más eficiente y robusto para grandes volúmenes de datos.")
    print("• One-Class SVM es útil cuando el espacio 'normal' es complejo, pero requiere más ajuste.")
    print("• F1 y ROC-AUC solo aplican cuando existen etiquetas de referencia.")

if __name__ == "__main__":
    main()
