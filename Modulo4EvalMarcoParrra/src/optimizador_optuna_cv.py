import optuna
import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from src.evaluador import evaluar_modelo_cv
from src.modelos import crear_modelo_random_forest

def optimizar_con_optuna_cv(X_train, y_train, n_trials, tuned_params):
    """
    Optimiza hiperparámetros con Optuna usando validación cruzada.
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
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    start = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    end = time.time()

    print("✅ Mejores parámetros Optuna:", study.best_params)

    best_model = crear_modelo_random_forest(**study.best_params)
    best_model.fit(X_train, y_train)

    resultado = evaluar_modelo_cv(
        "Optuna",
        best_model,
        X_train,
        y_train,
        end - start,
        study.best_params,
    )

    resultado["mejores_parametros"] = study.best_params
    return resultado