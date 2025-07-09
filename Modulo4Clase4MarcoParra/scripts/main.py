import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.utils import cargar_datos_breast_cancer, preprocesar_datos
from src.modelos import entrenar_modelo_base
from src.evaluador import evaluar_modelo
from src.optimizador import (
    optimizar_con_gridsearch,
    optimizar_con_randomsearch,
)
from src.optimizador_optuna import optimizar_con_optuna
from src.optimizador_skopt import optimizar_con_skopt
from src.optimizador_genetico import optimizar_con_genetico
from src.visualizador import (
    visualizar_matriz_confusion,
    visualizar_curva_roc,
    graficar_metricas_comparativas,
)

NUM_TRIALS = 200
tuned_params = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [5, 10, 15, 20, 30],
    "min_samples_split": [2, 3, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

if __name__ == "__main__":
    df = cargar_datos_breast_cancer()
    X_train, X_test, y_train, y_test = preprocesar_datos(df)
    resultados = []

    # Base
    resultado_base = entrenar_modelo_base(X_train, y_train, X_test, y_test)
    resultados.append(resultado_base)
    visualizar_matriz_confusion(y_test, resultado_base["y_pred"], metodo="Base")
    visualizar_curva_roc(y_test, resultado_base["y_prob"], metodo="Base")

    # Genético
    resultado_gen = optimizar_con_genetico(
        X_train, y_train, X_test, y_test, n_generaciones=NUM_TRIALS, tuned_params=tuned_params
    )
    resultados.append(resultado_gen)
    visualizar_matriz_confusion(y_test, resultado_gen["y_pred"], metodo="Genético")
    visualizar_curva_roc(y_test, resultado_gen["y_prob"], metodo="Genético")

    # Grid Search
    resultado_grid = optimizar_con_gridsearch(X_train, y_train, X_test, y_test, tuned_params)
    resultados.append(resultado_grid)
    visualizar_matriz_confusion(y_test, resultado_grid["y_pred"], metodo="GridSearch")
    visualizar_curva_roc(y_test, resultado_grid["y_prob"], metodo="GridSearch")

    # Random Search
    resultado_random = optimizar_con_randomsearch(X_train, y_train, X_test, y_test, NUM_TRIALS, tuned_params)
    resultados.append(resultado_random)
    visualizar_matriz_confusion(y_test, resultado_random["y_pred"], metodo="RandomSearch")
    visualizar_curva_roc(y_test, resultado_random["y_prob"], metodo="RandomSearch")

    # Optuna
    resultado_optuna = optimizar_con_optuna(X_train, y_train, X_test, y_test, NUM_TRIALS, tuned_params)
    resultados.append(resultado_optuna)
    visualizar_matriz_confusion(y_test, resultado_optuna["y_pred"], metodo="Optuna")
    visualizar_curva_roc(y_test, resultado_optuna["y_prob"], metodo="Optuna")

    # Skopt
    resultado_skopt = optimizar_con_skopt(X_train, y_train, X_test, y_test, NUM_TRIALS, tuned_params)
    resultados.append(resultado_skopt)
    visualizar_matriz_confusion(y_test, resultado_skopt["y_pred"], metodo="Skopt")
    visualizar_curva_roc(y_test, resultado_skopt["y_prob"], metodo="Skopt")

    graficar_metricas_comparativas(resultados)

    print("\n✅ Pipeline completado.")
