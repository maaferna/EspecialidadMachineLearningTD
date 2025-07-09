import optuna
import time
from sklearn.metrics import f1_score
from src.evaluador import evaluar_modelo
from src.modelos import crear_modelo_random_forest


def optimizar_con_optuna(X_train, y_train, X_test, y_test, n_trials, tuned_params):
    """
    Optimiza hiperparámetros con Optuna.
    """

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

    return evaluar_modelo(
        "Optuna",
        best_model,
        X_test,
        y_test,
        end - start,
        study.best_params,
    )
