# src/visualizador.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def graficar_coeficientes_elasticnet(model, feature_names):
    """
    Dibuja un gráfico de barras con los coeficientes del modelo ElasticNet.
    """
    coef = model.coef_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coef, y=feature_names)
    plt.title("Coeficientes del modelo ElasticNet")
    plt.xlabel("Peso")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig("outputs/coeficientes_elasticnet.png")
    plt.close()
    print("📊 Gráfico de coeficientes ElasticNet guardado en outputs/coeficientes_elasticnet.png")


def graficar_pinball_loss_quantiles():
    """
    Carga resultados de regresión cuantílica y grafica el pinball loss para cada cuantil.
    """
    df = pd.read_csv("outputs/quantile_resultados.csv")

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="quantile", y="pinball_loss", marker="o")
    plt.title("Pinball Loss por Cuantil")
    plt.xlabel("Cuantil")
    plt.ylabel("Pinball Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/quantile_pinball_loss.png")
    plt.close()
    print("📉 Gráfico pinball loss por cuantil guardado en outputs/quantile_pinball_loss.png")


def graficar_forecast_var():
    """
    Carga el forecast de VAR y lo grafica como líneas por variable.
    """
    df = pd.read_csv("outputs/var_forecast.csv")

    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], marker="o", label=column)
    plt.title("Forecast VAR a 5 pasos")
    plt.xlabel("Paso")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/var_forecast_plot.png")
    plt.close()
    print("📈 Forecast VAR graficado en outputs/var_forecast_plot.png")


