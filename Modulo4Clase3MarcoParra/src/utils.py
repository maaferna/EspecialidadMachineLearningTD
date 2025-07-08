"""
utils.py - Funciones auxiliares para carga de datos y entrenamiento base
"""

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



def cargar_datos():
    """Carga el dataset de cáncer de mama y preparar para el entrenamiento"""
    data = load_breast_cancer()
    X, y = data.data, data.target

    print(X.shape, y.shape)  # Verifica las dimensiones de los datos
    print(X[:5], y[:5])  # ✅ Esto mostrará las primeras 5 filas


    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División 70/30
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)


def entrenar_modelo_base(X_train, y_train, X_test, y_test):
    """Entrena un modelo RandomForest sin optimización"""
    start = time.time()
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    end = time.time()

    f1 = f1_score(y_test, y_pred)
    print("📋 Classification Report (Base):")
    print(classification_report(y_test, y_pred))
    print(f"🎯 F1-Score (Base): {f1:.4f}")

    return {
        "metodo": "base",
        "f1": f1,
        "tiempo": end - start
    }


def optimizar_con_skopt(X_train, y_train, X_test, y_test):
    """Aplica Scikit-Optimize para encontrar los mejores hiperparámetros"""
    search_space = {
        "n_estimators": Integer(10, 500), # Número de árboles en el bosque
        "max_depth": Integer(2, 20), # Profundidad máxima de los árboles
        "min_samples_split": Integer(2, 10) # Mínimo número de muestras para dividir un nodo
    }

    model = RandomForestClassifier(random_state=42)

    opt = BayesSearchCV(
        model,
        search_spaces=search_space,
        cv=3, # Validación cruzada de 3 pliegues
        n_iter=100,  # Número de iteraciones de optimización
        verbose=0,  # Silenciar salida detallada
        scoring="f1", # Métrica de evaluación
        random_state=42,
        n_jobs=-1 # Usar todos los núcleos disponibles
    )

    print("🔍 Iniciando búsqueda bayesiana con Scikit-Optimize...")
    start = time.time()
    opt.fit(X_train, y_train)
    end = time.time()

    best_model = opt.best_estimator_
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print("✅ Mejores parámetros encontrados:", opt.best_params_)
    print("📋 Classification Report (Skopt):")
    print(classification_report(y_test, y_pred))
    print(f"🎯 F1-Score: {f1:.4f}")
    print(f"⏱️ Tiempo de optimización: {end - start:.2f} segundos")

    return {
        "metodo": "skopt",
        "f1": f1,
        "tiempo": end - start,
        "evolucion": opt.cv_results_["mean_test_score"].tolist()
    }



def optimizar_con_hyperopt(X_train, y_train, X_test, y_test, max_evals=100):
    """Optimiza hiperparámetros usando Hyperopt y RandomForestClassifier"""
    scores_evolucion = []  # Lista para almacenar la evolución de las F1-scores
    def objective(params):
        """Función objetivo para Hyperopt"""
        clf = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            random_state=42
        )
        f1 = cross_val_score(clf, X_train, y_train, cv=3, scoring="f1").mean()
        
        scores_evolucion.append(f1)
        return {"loss": -f1, "status": STATUS_OK}

    # Definir espacio de búsqueda
    space = {
        "n_estimators": hp.quniform("n_estimators", 10, 500, 1),
        "max_depth": hp.quniform("max_depth", 2, 20, 1),
        "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1)
    } 

    print("🔍 Iniciando búsqueda bayesiana con Hyperopt...")
    start = time.time()
    trials = Trials() # Objeto que sirve para almacenar el historial completo de las ejecuciones
    best = fmin(
        fn=objective,  # Función objetivo a minimizar
        space=space, # Espacio de búsqueda de hiperparámetros
        algo=tpe.suggest, # Algoritmo de optimización
        max_evals=max_evals, 
        trials=trials,
        rstate= np.random.default_rng(42)  # usa Generator con seed reproducible

    )
    end = time.time()

    # Convertir parámetros a enteros
    best = {k: int(v) for k, v in best.items()}

    print("✅ Mejores parámetros encontrados:", best)

    # Entrenar modelo final con mejores parámetros
    clf_final = RandomForestClassifier(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        min_samples_split=best["min_samples_split"],
        random_state=42
    )
    clf_final.fit(X_train, y_train)
    y_pred = clf_final.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print("📋 Classification Report (Hyperopt):")
    print(classification_report(y_test, y_pred))
    print(f"🎯 F1-Score: {f1:.4f}")
    print(f"⏱️ Tiempo de optimización: {end - start:.2f} segundos")

    return {
        "metodo": "hyperopt",
        "f1": f1,
        "tiempo": end - start,
        "evolucion": scores_evolucion
    }

