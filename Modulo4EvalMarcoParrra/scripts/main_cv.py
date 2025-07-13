import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import preprocesar_datos_multiclase_cv
from src.visualizador_cv import visualizar_curva_roc, graficar_metricas_comparativas
from src.optimizador_ray_cv import optimizar_con_raytune_cv
from src.optimizador_genetico_cv import  optimizar_con_genetico_cv
from src.optimizador_cv import optimizar_con_gridsearch_cv, optimizar_con_randomsearch_cv
from src.optimizador_optuna_cv import optimizar_con_optuna_cv
from src.optimizador_skopt_cv import optimizar_con_hyperopt_cv, optimizar_con_skopt_cv
from src.modelos import crear_modelo_random_forest, entrenar_modelo_base
from src.evaluador import evaluar_modelo, evaluar_modelo_cv

print("\n📦 Cargando datos...")
train_df = pd.read_csv("data/Training.csv")
test_df = pd.read_csv("data/Testing.csv")

X_train, y_train, scaler, label_encoder = preprocesar_datos_multiclase_cv(train_df, fit_scaler=True)
X_test, y_test, _, _ = preprocesar_datos_multiclase_cv(test_df, scaler=scaler, label_encoder=label_encoder)

tuned_params = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [5, 10, 15, 20, 30],
    "min_samples_split": [2, 3, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

NUM_TRIALS = 10
resultados = []


resultado_ray = optimizar_con_raytune_cv(X_train, y_train, NUM_TRIALS, tuned_params)
resultados.append(resultado_ray)
modelo_ray = crear_modelo_random_forest(**resultado_ray["mejores_parametros"])
modelo_ray.fit(X_train, y_train)
resultado_test_ray = evaluar_modelo("RayTune", modelo_ray, X_test, y_test, 0.0, resultado_ray["mejores_parametros"])
resultados.append(resultado_test_ray)
visualizar_curva_roc(y_test, resultado_test_ray["y_prob"], metodo="RayTune") 


resultado_base = entrenar_modelo_base(X_train, y_train, X_test, y_test)
resultados.append(resultado_base)
visualizar_curva_roc(y_test, resultado_base["y_prob"], metodo="Base")


resultado_gen = optimizar_con_genetico_cv(X_train, y_train, NUM_TRIALS, tuned_params)
modelo_gen = crear_modelo_random_forest(**resultado_gen["mejores_parametros"])
modelo_gen.fit(X_train, y_train)
resultado_test_gen = evaluar_modelo("Genético", modelo_gen, X_test, y_test, 0.0, resultado_gen["mejores_parametros"])
resultados.append(resultado_test_gen)
visualizar_curva_roc(y_test, resultado_test_gen["y_prob"], metodo="Genético")



'''

resultado_grid = optimizar_con_gridsearch_cv(X_train, y_train, tuned_params)
modelo_grid = crear_modelo_random_forest(**resultado_grid["mejores_parametros"])
modelo_grid.fit(X_train, y_train)
resultado_test_grid = evaluar_modelo("GridSearch", modelo_grid, X_test, y_test, 0.0, resultado_grid["mejores_parametros"])
resultados.append(resultado_test_grid)
visualizar_curva_roc(y_test, resultado_test_grid["y_prob"], metodo="GridSearch")


resultado_optuna = optimizar_con_optuna_cv(X_train, y_train, NUM_TRIALS, tuned_params)
modelo_optuna = crear_modelo_random_forest(**resultado_optuna["mejores_parametros"])
modelo_optuna.fit(X_train, y_train)
resultado_test_optuna = evaluar_modelo("Optuna", modelo_optuna, X_test, y_test, 0.0, resultado_optuna["mejores_parametros"])
resultados.append(resultado_test_optuna)
visualizar_curva_roc(y_test, resultado_test_optuna["y_prob"], metodo="Optuna")

resultado_skopt = optimizar_con_skopt_cv(X_train, y_train, NUM_TRIALS, tuned_params)
modelo_skopt = crear_modelo_random_forest(**resultado_skopt["mejores_parametros"])
modelo_skopt.fit(X_train, y_train)
resultado_test_skopt = evaluar_modelo("Skopt", modelo_skopt, X_test, y_test, 0.0, resultado_skopt["mejores_parametros"])
resultados.append(resultado_test_skopt)
visualizar_curva_roc(y_test, resultado_test_skopt["y_prob"], metodo="Skopt")

resultado_hyperopt = optimizar_con_hyperopt_cv(X_train, y_train, NUM_TRIALS, tuned_params)
modelo_hyperopt = crear_modelo_random_forest(**resultado_hyperopt["mejores_parametros"])
modelo_hyperopt.fit(X_train, y_train)
resultado_test_hyperopt = evaluar_modelo("Hyperopt", modelo_hyperopt, X_test, y_test, 0.0, resultado_hyperopt["mejores_parametros"])
resultados.append(resultado_test_hyperopt)
visualizar_curva_roc(y_test, resultado_test_hyperopt["y_prob"], metodo="Hyperopt")

resultado_random = optimizar_con_randomsearch_cv(X_train, y_train, NUM_TRIALS, tuned_params)
modelo_random = crear_modelo_random_forest(**resultado_random["mejores_parametros"])
modelo_random.fit(X_train, y_train)
resultado_test_random = evaluar_modelo("RandomSearch", modelo_random, X_test, y_test, 0.0, resultado_random["mejores_parametros"])
resultados.append(resultado_test_random)
visualizar_curva_roc(y_test, resultado_test_random["y_prob"], metodo="RandomSearch")

'''


graficar_metricas_comparativas(resultados)

print("\n📈 Pipeline completado.")
