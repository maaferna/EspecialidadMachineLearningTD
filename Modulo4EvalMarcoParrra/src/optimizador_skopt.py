
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from skopt import BayesSearchCV
from skopt.space import Integer
import time
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
import numpy as np


from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from src.evaluador import evaluar_modelo
from src.modelos import crear_modelo_random_forest





def optimizar_con_skopt(X_train, y_train, X_test, y_test, n_trials, tuned_params):
    """
    Optimiza hiperparámetros con Bayesian Optimization (skopt).
    Adaptado para clasificación multiclase.
    """
    print("\n🧠 Optimizando con BayesSearchCV (skopt)...")
    start = time.time()

    model = crear_modelo_random_forest()
    bayes = BayesSearchCV(
        estimator=model,
        search_spaces=tuned_params,
        n_iter=n_trials,
        cv=3,
        scoring="f1_macro",  # ✅ clave para multiclase
        n_jobs=-1,
        random_state=42,
    )

    bayes.fit(X_train, y_train)
    end = time.time()

    print("✅ Mejores parámetros skopt:", bayes.best_params_)
    return evaluar_modelo(
        "Skopt",
        bayes.best_estimator_,
        X_test,
        y_test,
        end - start,
        bayes.best_params_,
    )



def optimizar_con_hyperopt(X_train, y_train, X_test, y_test, n_trials, tuned_params):
    """
    Optimiza hiperparámetros usando Hyperopt y RandomForestClassifier.
    Adaptado a clasificación multiclase.
    """
    print("\n🧬 Optimizando con Hyperopt...")

    scores_evolucion = []

    # Convertir espacio de búsqueda al formato de Hyperopt
    space = {
        "n_estimators": hp.choice("n_estimators", tuned_params["n_estimators"]),
        "max_depth": hp.choice("max_depth", tuned_params["max_depth"]),
        "min_samples_split": hp.choice("min_samples_split", tuned_params["min_samples_split"]),
    }

    def objective(params):
        model = crear_modelo_random_forest(**params)
        f1 = cross_val_score(
            model,
            X_train,
            y_train,
            cv=3,
            scoring="f1_macro",  # ✅ multiclase
            n_jobs=-1,
        ).mean()
        scores_evolucion.append(f1)
        return {"loss": -f1, "status": STATUS_OK}

    start = time.time()
    trials = Trials()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials,
        rstate=np.random.default_rng(42),
    )
    end = time.time()

    # Reconstruir mejores parámetros desde índices elegidos
    best_params = {
        "n_estimators": tuned_params["n_estimators"][best["n_estimators"]],
        "max_depth": tuned_params["max_depth"][best["max_depth"]],
        "min_samples_split": tuned_params["min_samples_split"][best["min_samples_split"]],
    }

    print("✅ Mejores parámetros Hyperopt:", best_params)

    best_model = crear_modelo_random_forest(**best_params)
    best_model.fit(X_train, y_train)

    return evaluar_modelo(
        "Hyperopt",
        best_model,
        X_test,
        y_test,
        end - start,
        best_params,
    )
