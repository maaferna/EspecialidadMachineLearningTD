# src/evaluador.py

import pandas as pd
from sklearn.metrics import mean_squared_error


def evaluar_modelo(modelo, X_train, y_train, X_test, y_test):
    """
    Evalúa un modelo de regresión utilizando el conjunto de entrenamiento y prueba.

    Args:
        modelo: Modelo de regresión a evaluar.
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns:
        dict: Diccionario con el modelo entrenado, MSE y predicciones.
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "modelo": modelo,
        "mse": mse,
        "predicciones": y_pred
    }


def buscar_mejor_modelo(nombre_modelo, config, X_train, y_train, X_test, y_test):
    """
    Busca el mejor modelo de regresión según los parámetros especificados.

    Args:
        nombre_modelo (str): Nombre del modelo a evaluar.
        config (dict): Configuración del modelo con función de creación y parámetros.
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Variable objetivo de prueba.

    Returns
        dict: Diccionario con el mejor modelo, MSE y parámetros.
    """
    mejor_mse = float("inf")
    mejor_params = None
    resultados_intermedios = []

    for idx, params in enumerate(config["params"]):
        modelo = config["model_fn"](**params)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        resultados_intermedios.append({
            "modelo": nombre_modelo,
            "instancia": idx,
            "mse": mse,
            "parametros": str(params)
        })

        if mse < mejor_mse:
            mejor_mse = mse
            mejor_params = params

    # Guardar resultados parciales (se concatena si ya existe)
    try:
        df_existente = pd.read_csv("./outputs/todas_las_instancias.csv")
        df_nuevo = pd.DataFrame(resultados_intermedios)
        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
    except FileNotFoundError:
        df_final = pd.DataFrame(resultados_intermedios)

    df_final.to_csv("./outputs/todas_las_instancias.csv", index=False)

    return {
        "modelo": nombre_modelo,
        "mejor_mse": mejor_mse,
        "parametros": mejor_params
    }