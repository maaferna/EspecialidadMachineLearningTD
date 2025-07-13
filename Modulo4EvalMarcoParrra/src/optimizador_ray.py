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
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------
# Ajustar PYTHONPATH dentro del worker de Ray
# -----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



from src.evaluador import evaluar_modelo  # noqa: E402  pylint: disable=C0413
from src.modelos import crear_modelo_random_forest  # noqa: E402  pylint: disable=C0413

# -----------------------------------------------------------------------------
# Ray Tune + OptunaSearch para clasificación multiclase
# -----------------------------------------------------------------------------
# * F1-score macro como métrica principal.
# * Captura de excepciones por trial para evitar errores globales.
# * Ajuste de PYTHONPATH para que los workers importen módulos locales.
# -----------------------------------------------------------------------------


def train_ray_tune(config: Dict[str, Any],
                   X_train, X_test, y_train, y_test) -> None:
    """Función ejecutada por cada trial de Ray Tune.

    - Crea un modelo RandomForest con los hiperparámetros indicados.
    - Calcula `f1_macro` sobre el conjunto de prueba.
    - Reporta la métrica a Ray Tune.
    - En caso de excepción se reporta `f1_macro = 0.0` para que el experimento
      continúe sin marcar el trial como fallido.
    """
    f1_macro = 0.0
    try:
        model = crear_modelo_random_forest(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average="macro")
    except Exception as exc:
        print(f"[RayTune Trial Error]: {exc}")

    session.report({"f1_macro": f1_macro})



def optimizar_con_raytune(
    X_train,
    y_train,
    X_test,
    y_test,
    n_trials: int,
    tuned_params: Dict[str, Any],
):
    """Optimiza hiperparámetros con Ray Tune + OptunaSearch.«"""  # noqa: D401

    print("\n⚡ Optimizando con Ray Tune (OptunaSearch)...")
    start = time.time()

    # --------------------------- Espacio de búsqueda -------------------------
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
            train_ray_tune,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        ),
        config=search_space,
        num_samples=n_trials,
        scheduler=scheduler,
        search_alg=algo,
        progress_reporter=reporter,
        resources_per_trial={"cpu": 1},
        verbose=1,
        fail_fast="raise",  # Interrumpe la ejecución si hay error sistemático
    )

    best_config = analysis.get_best_config(metric=metric_name, mode=mode)
    print("✅ Mejores parámetros Ray Tune:", best_config)

    best_model = crear_modelo_random_forest(**best_config)
    best_model.fit(X_train, y_train)

    end = time.time()

    return evaluar_modelo(
        "RayTune",
        best_model,
        X_test,
        y_test,
        end - start,
        best_config,
    )
