import time
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from src.evaluador import evaluar_modelo
from src.modelos import crear_modelo_random_forest
import numpy as np


def train_ray_tune(config, X_train, X_test, y_train, y_test):
    """Función de entrenamiento que Ray Tune ejecutará para cada configuración."""
    model = crear_modelo_random_forest(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    tune.report(f1=f1)


def optimizar_con_raytune(X_train, y_train, X_test, y_test, n_trials, tuned_params):
    """
    Optimización de hiperparámetros con Ray Tune + Optuna.
    """
    print("\n⚡ Optimizando con Ray Tune (OptunaSearch)...")
    start = time.time()

    search_space = {
        "n_estimators": tune.choice(tuned_params["n_estimators"]),
        "max_depth": tune.choice(tuned_params["max_depth"]),
        "min_samples_split": tune.choice(tuned_params["min_samples_split"]),
    }

    algo = OptunaSearch()
    scheduler = ASHAScheduler(metric="f1", mode="max")

    reporter = CLIReporter(metric_columns=["f1", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_ray_tune,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        ),
        resources_per_trial={"cpu": 1},
        config=search_space,
        num_samples=n_trials,
        scheduler=scheduler,
        search_alg=algo,
        progress_reporter=reporter,
        verbose=1,
    )

    end = time.time()
    best_config = analysis.get_best_config(metric="f1", mode="max")
    print("✅ Mejores parámetros Ray Tune:", best_config)

    best_model = crear_modelo_random_forest(**best_config)
    best_model.fit(X_train, y_train)

    return evaluar_modelo(
        "RayTune",
        best_model,
        X_test,
        y_test,
        end - start,
        best_config,
    )
