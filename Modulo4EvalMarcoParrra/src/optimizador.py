import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.modelos import crear_modelo_random_forest
from src.evaluador import evaluar_modelo


def optimizar_con_gridsearch(X_train, y_train, X_test, y_test, tuned_params):
    """
    Realiza una búsqueda de hiperparámetros usando Grid Search.
    Devuelve un diccionario con las métricas del modelo optimizado.
    """
    print("\n🔧 Grid Search en progreso...")
    start = time.time()
    model = crear_modelo_random_forest()
    grid = GridSearchCV(
        model, tuned_params, cv=3, scoring="f1_macro", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    end = time.time()

    print("✅ Mejores parámetros Grid Search:", grid.best_params_)
    return evaluar_modelo(
        "GridSearch", grid.best_estimator_,
        X_test, y_test, end - start, grid.best_params_
    )


def optimizar_con_randomsearch(X_train, y_train, X_test, y_test, n_trials, tuned_params):
    """
    Realiza una búsqueda de hiperparámetros usando Random Search.
    Devuelve un diccionario con las métricas del modelo optimizado.
    """
    print("\n🍀 Random Search en progreso...")
    start = time.time()
    model = crear_modelo_random_forest()
    random_search = RandomizedSearchCV(
        model,
        param_distributions=tuned_params,
        n_iter=n_trials,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)
    end = time.time()

    print("✅ Mejores parámetros Random Search:", random_search.best_params_)

    
    return evaluar_modelo(
        "RandomSearch", random_search.best_estimator_,
        X_test, y_test, end - start, random_search.best_params_
    )
