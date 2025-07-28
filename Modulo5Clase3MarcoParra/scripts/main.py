# scripts/main.py

from src.visualizador import graficar_accuracy_mejores_modelos, graficar_matriz_confusion, graficar_todos_los_resultados
from src.utils import cargar_y_preprocesar_adult_income
from src.modelos import (
    entrenar_random_forest,
    entrenar_adaboost,
    entrenar_xgboost,
    entrenar_lightgbm,
    entrenar_catboost
)
import warnings
warnings.filterwarnings("ignore")

def main():
    print("🔹 Cargando y preprocesando el dataset Adult Income...")
    X_train, X_test, y_train, y_test = cargar_y_preprocesar_adult_income()
    print(f"✅ Dataset dividido -> X_train: {X_train.shape}, X_test: {X_test.shape}")

   # Entrenar modelos y recolectar resultados
    resultados = []
    resultados.append(entrenar_random_forest(X_train, y_train, X_test, y_test))
    resultados.append(entrenar_adaboost(X_train, y_train, X_test, y_test))
    resultados.append(entrenar_xgboost(X_train, y_train, X_test, y_test))
    resultados.append(entrenar_lightgbm(X_train, y_train, X_test, y_test))
    resultados.append(entrenar_catboost(X_train, y_train, X_test, y_test))

    graficar_accuracy_mejores_modelos()
    graficar_todos_los_resultados()
    for modelo in ["RandomForest", "AdaBoost", "XGBoost", "LightGBM", "CatBoost"]:
        graficar_matriz_confusion(modelo)

    print("✅ Comparación de modelos completada.")

if __name__ == "__main__":
    main()
