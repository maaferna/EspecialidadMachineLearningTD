import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from statsmodels.tsa.api import VAR


def entrenar_elasticnet(X_train, y_train, X_test, y_test, grid_params):

    """Entrena un modelo ElasticNet con los parámetros especificados y evalúa su rendimiento.

    Args:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Variable objetivo de prueba.
        grid_params (list): Lista de diccionarios con los parámetros a evaluar.

    Returns:
        None
    """
    resultados = []

    for params in grid_params:
        model = ElasticNet(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        resultados.append({
            "alpha": params["alpha"],
            "l1_ratio": params["l1_ratio"],
            "mse": mse
        })

        print(f"🔧 ElasticNet | alpha: {params['alpha']}, l1_ratio: {params['l1_ratio']} -> MSE: {mse:.4f}")

    best_result = min(resultados, key=lambda x: x["mse"])
    print(f"🏆 Mejor ElasticNet -> alpha={best_result['alpha']}, l1_ratio={best_result['l1_ratio']}, MSE={best_result['mse']:.4f}")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("outputs/elasticnet_resultados.csv", index=False)
    df_resultados[df_resultados["mse"] == best_result["mse"]].to_csv("outputs/elasticnet_mejor.csv", index=False)

    best_model = ElasticNet(alpha=best_result["alpha"], l1_ratio=best_result["l1_ratio"])
    best_model.fit(X_train, y_train)
    return best_model



def entrenar_regresion_cuantilica(X_train, y_train, X_test, y_test, quantiles):
    """Entrena un modelo de regresión cuantílica para diferentes cuantiles y evalúa su rendimiento.

    Args:
        X_train (pd.DataFrame): Conjunto de características de entrenamiento.
        y_train (pd.Series): Variable objetivo de entrenamiento.
        X_test (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Variable objetivo de prueba.
        quantiles (list): Lista de cuantiles a evaluar.

    Returns:
        None
    """
    resultados = []

    for q in quantiles:
        model = QuantileRegressor(quantile=q, solver="highs")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        pinball = mean_pinball_loss(y_test, y_pred, alpha=q)
        resultados.append({"quantile": q, "pinball_loss": pinball})
        print(f"🔧 QuantileRegressor (q={q:.1f}) -> Pinball loss: {pinball:.4f}")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("outputs/quantile_resultados.csv", index=False)
    best_result = df_resultados.loc[df_resultados["pinball_loss"].idxmin()]
    df_resultados[df_resultados["pinball_loss"] == best_result["pinball_loss"]].to_csv("outputs/quantile_mejor.csv", index=False)



def entrenar_var(df, var_config):
    """Entrena un modelo VAR (Vector Autoregression) y muestra el pronóstico.

    Args:
        df (pd.DataFrame): DataFrame con las series temporales.
        var_config (dict): Configuración del modelo VAR, incluyendo el número máximo de lags.

    Returns:
        None
    """
    model = VAR(df)
    selected_lag = model.select_order(var_config["maxlags"])
    best_lag = selected_lag.aic
    print(f"📌 Lag óptimo seleccionado (AIC): {best_lag}")

    var_model = model.fit(best_lag)
    print(var_model.summary())

    forecast = var_model.forecast(df.values[-best_lag:], steps=5)
    forecast_df = pd.DataFrame(forecast, columns=df.columns)
    print("📈 Pronóstico a 5 pasos:")
    print(forecast_df)

    forecast_df.to_csv("outputs/var_forecast.csv", index=False)


