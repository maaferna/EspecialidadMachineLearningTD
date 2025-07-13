import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from ray import tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from sklearn.model_selection import cross_val_score

from src.modelos import crear_modelo_random_forest
from src.evaluador import evaluar_modelo_cv

# Ajustar PYTHONPATH dentro del worker de Ray
# -----------------------------------------------------------------------------
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def train_ray_tune_cv(config: Dict[str, Any], X_train, y_train) -> None:
    """
    Función ejecutada por cada trial de Ray Tune para validación cruzada.
    """
    f1_macro = 0.0
    try:
        model = crear_modelo_random_forest(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
        )
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1)
        f1_macro = scores.mean()
    except Exception as exc:
        print(f"[RayTune Trial Error]: {exc}")
    session.report({"f1_macro": f1_macro})

def optimizar_con_raytune_cv(
    X_train,
    y_train,
    n_trials: int,
    tuned_params: Dict[str, Any],
):
    """
    Optimiza hiperparámetros con Ray Tune + OptunaSearch usando validación cruzada.
    """
    print("\n⚡ Optimizando con Ray Tune (OptunaSearch)...")
    start = time.time()

    search_space = {
        "n_estimators": tune.choice(tuned_params["n_estimators"]),
        "max_depth": tune.choice(tuned_params["max_depth"]),
        "min_samples_split": tune.choice(tuned_params["min_samples_split"]),
    }

    metric_name = "f1_macro"
    mode = "max"

    algo = OptunaSearch(metric=metric_name, mode=mode)
    scheduler = ASHAScheduler(metric=metric_name, mode=mode)
    reporter = CLIReporter(metric_columns=[metric_name, "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_ray_tune_cv,
            X_train=X_train,
            y_train=y_train,
        ),
        config=search_space,
        num_samples=n_trials,
        scheduler=scheduler,
        search_alg=algo,
        progress_reporter=reporter,
        resources_per_trial={"cpu": 1},
        verbose=1,
        fail_fast="raise",
    )

    best_config = analysis.get_best_config(metric=metric_name, mode=mode)
    print("✅ Mejores parámetros Ray Tune:", best_config)

    # Entrenar modelo final con mejores parámetros sobre todo el train
    best_model = crear_modelo_random_forest(**best_config)
    best_model.fit(X_train, y_train)

    end = time.time()

    # Calcular F1 macro final CV (puedes repetir cross_val_score aquí si quieres mostrarlo)
    f1_cv = cross_val_score(best_model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
    f1_cv_mean = f1_cv.mean()
    f1_cv_std = f1_cv.std()

    return {
        "metodo": "RayTune",
        "mejores_parametros": best_config,
        "f1_cv": f1_cv_mean,
        "f1_cv_std": f1_cv_std,
        "modelo": best_model,
        "tiempo": end - start,
        # Opcional: puedes agregar otros resultados aquí
    }
