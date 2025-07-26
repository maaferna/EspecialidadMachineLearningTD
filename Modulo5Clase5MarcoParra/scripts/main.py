# scripts/main.py

import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from src.visualizador import graficar_coeficientes_modelo, graficar_mejores_resultados, graficar_todas_las_instancias
from src.utils import cargar_dataset, preprocesar_datos
from src.modelos import crear_modelo_lasso, crear_modelo_ridge, crear_modelo_elasticnet
from src.evaluador import buscar_mejor_modelo

warnings.filterwarnings("ignore")

param_grid = {
    "lasso": {
        "model_fn": crear_modelo_lasso,
        "params": [
            {"alpha": 0.001},
            {"alpha": 0.01},
            {"alpha": 0.1},
            {"alpha": 1.0},
        ],
    },
    "ridge": {
        "model_fn": crear_modelo_ridge,
        "params": [
            {"alpha": 0.001},
            {"alpha": 0.01},
            {"alpha": 0.1},
            {"alpha": 1.0},
        ],
    },
    "elasticnet": {
        "model_fn": crear_modelo_elasticnet,
        "params": [
            {"alpha": 0.1, "l1_ratio": 0.2},
            {"alpha": 0.1, "l1_ratio": 0.5},
            {"alpha": 0.1, "l1_ratio": 0.8},
            {"alpha": 1.0, "l1_ratio": 0.5},
        ],
    },
}

def main():
    # Cargar dataset
    df = cargar_dataset("data/Fish.csv")

    # Preprocesar
    X, y, scaler = preprocesar_datos(df)

    print("\n✅ Preprocesamiento completado.")
    print(f"📐 X shape: {X.shape}")
    print(f"🎯 y shape: {y.shape}")

    print("\n🔎 Primeras filas de X (features):")
    print(X.head())

    print("\n🎯 Primeros valores de y (target):")
    print(y.head())

    print("\n📏 Scaler utilizado para estandarización:")
    print(scaler)

    # Separar datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluar cada modelo
    resultados = []
    for nombre_modelo, config in param_grid.items():
        resultado = buscar_mejor_modelo(nombre_modelo, config, X_train, y_train, X_test, y_test)
        print(f"\n✅ Mejor resultado para {nombre_modelo.upper()}: MSE={resultado['mejor_mse']:.4f} con {resultado['parametros']}")
        resultados.append(resultado)
        # Entrenar con mejores parámetros para visualizar coeficientes
        modelo_entrenado = config["model_fn"](**resultado["parametros"])
        modelo_entrenado.fit(X_train, y_train)
        graficar_coeficientes_modelo(modelo_entrenado, X.columns, nombre_modelo)

    # Guardar resultados
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("outputs/resultados_gridsearch.csv", index=False)
    print("\n📦 Resultados exportados a outputs/resultados_gridsearch.csv")

    graficar_mejores_resultados()
    graficar_todas_las_instancias()

if __name__ == "__main__":
    main()
