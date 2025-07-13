import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.modelos import crear_modelo_random_forest
from src.evaluador import evaluar_modelo_cv  # ⚠️ asegúrate de importar la versión correcta


def optimizar_con_gridsearch_cv(X, y, tuned_params):
    """
    Optimización de hiperparámetros con GridSearchCV y validación cruzada.
    """
    print("\n🔧 Grid Search en progreso (CV)...")
    start = time.time()
    model = crear_modelo_random_forest()
    grid = GridSearchCV(
        model, tuned_params, cv=5, scoring="f1_macro", n_jobs=-1
    )
    grid.fit(X, y)
    end = time.time()

    print("✅ Mejores parámetros Grid Search:", grid.best_params_)
    return evaluar_modelo_cv(
        "GridSearch",
        grid.best_estimator_,
        X,
        y,
        end - start,
        grid.best_params_,
    )


def optimizar_con_randomsearch_cv(X, y, n_trials, tuned_params):
    """
    Optimización de hiperparámetros con RandomizedSearchCV y validación cruzada.
    """
    print("\n🍀 Random Search en progreso (CV)...")
    start = time.time()
    model = crear_modelo_random_forest()
    random_search = RandomizedSearchCV(
        model,
        param_distributions=tuned_params,
        n_iter=n_trials,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    random_search.fit(X, y)
    end = time.time()

    print("✅ Mejores parámetros Random Search:", random_search.best_params_)
    resultado = evaluar_modelo_cv(
        "RandomSearch",
        random_search.best_estimator_,
        X,
        y,
        end - start,
        random_search.best_params_,
    )
    resultado["mejores_parametros"] = random_search.best_params_
    return resultado