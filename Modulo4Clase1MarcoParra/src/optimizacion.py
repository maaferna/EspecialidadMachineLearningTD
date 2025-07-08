import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
import time
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from src.modelos import crear_modelo_random_forest
from src.evaluador import evaluar_modelo


NUM_TRIALS = 500  # Hacerlo constante para todos los métodos

# Espacio de búsqueda unificado para todos los métodos
tuned_params = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
}

def optimizar_con_gridsearch(X_train, y_train, X_test, y_test):
    '''
    Realiza una búsqueda de hiperparámetros usando Grid Search.
    Devuelve un diccionario con las métricas del modelo optimizado.
    
    Parámetros:
    - X_train: Datos de entrenamiento.
    - y_train: Etiquetas de entrenamiento.
    - X_test: Datos de prueba.
    - y_test: Etiquetas de prueba.
    
    Retorna:
    - Diccionario con métricas del modelo optimizado.
    '''
    '''
    Realiza una búsqueda de hiperparámetros usando Grid Search.
    Devuelve un diccionario con las métricas del modelo optimizado.
    
    '''
    print("\n🔧 Grid Search en progreso...")
    start = time.time()
    model = crear_modelo_random_forest()
    grid = GridSearchCV(model, tuned_params, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    end = time.time()

    print("✅ Mejores parámetros Grid Search:", grid.best_params_)
    return evaluar_modelo("GridSearch", grid.best_estimator_, 
                          X_test, y_test, end - start, grid.best_params_)

def optimizar_con_randomsearch(X_train, y_train, X_test, y_test, n_iter=NUM_TRIALS):
    '''
    Realiza una búsqueda de hiperparámetros usando Random Search.
    Devuelve un diccionario con las métricas del modelo optimizado.
    
    Parámetros:
    - X_train: Datos de entrenamiento.
    - y_train: Etiquetas de entrenamiento.
    - X_test: Datos de prueba.
    - y_test: Etiquetas de prueba.
    - n_iter: Número de iteraciones para Random Search (por defecto, igual a NUM_TRIALS).
    
    Retorna:
    - Diccionario con métricas del modelo optimizado.
    '''
    print("\n🍀 Random Search en progreso...")
    start = time.time()
    model = crear_modelo_random_forest()
    random_search = RandomizedSearchCV(
        model, param_distributions=tuned_params, n_iter=n_iter, cv=3, scoring="f1", n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    end = time.time()

    print("✅ Mejores parámetros Random Search:", random_search.best_params_)
    return evaluar_modelo("RandomSearch", random_search.best_estimator_,
                           X_test, y_test, end - start, random_search.best_params_)

def optimizar_con_optuna(X_train, y_train, X_test, y_test, n_trials=NUM_TRIALS):
    '''
    Realiza una búsqueda de hiperparámetros usando Optuna.
    Devuelve un diccionario con las métricas del modelo optimizado.
    
    parámetros:
    - X_train: Datos de entrenamiento.
    - y_train: Etiquetas de entrenamiento.
    - X_test: Datos de prueba.
    - y_test: Etiquetas de prueba.
    - n_trials: Número de pruebas para Optuna (por defecto, igual a NUM_TRIALS).
    
    Retorna:
    - Diccionario con métricas del modelo optimizado.
    '''
    print("\n🔮 Optimizando con Optuna...")

    def objective(trial):
        n_estimators = trial.suggest_categorical("n_estimators", tuned_params["n_estimators"])
        max_depth = trial.suggest_categorical("max_depth", tuned_params["max_depth"])
        min_samples_split = trial.suggest_categorical("min_samples_split", tuned_params["min_samples_split"])

        model = crear_modelo_random_forest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return f1_score(y_test, y_pred)

    start = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    end = time.time()

    print("✅ Mejores parámetros Optuna:", study.best_params)

    best_model = crear_modelo_random_forest(**study.best_params)
    best_model.fit(X_train, y_train)

    return evaluar_modelo("Optuna", best_model, X_test, 
                          y_test, end - start, study.best_params)
