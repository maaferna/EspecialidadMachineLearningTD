import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def graficar_mejores_resultados(path_csv="./outputs/resultados_gridsearch.csv"):
    """
    Lee el CSV generado y grafica el mejor MSE por modelo.
    """
    df = pd.read_csv(path_csv)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="modelo", y="mejor_mse", data=df, palette="Set2")
    plt.title("📉 Mejor MSE por Modelo")
    plt.ylabel("MSE")
    plt.xlabel("Modelo")
    plt.tight_layout()
    plt.savefig("outputs/grafico_mejores_modelos.png")
    plt.show()


def graficar_todas_las_instancias(path_csv="./outputs/todas_las_instancias.csv"):
    """
    Grafica todas las combinaciones de parámetros evaluadas por cada modelo.
    """
    df = pd.read_csv(path_csv)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="instancia", y="mse", hue="modelo", marker="o")
    plt.title("📊 MSE por Configuración de Parámetros")
    plt.xlabel("Instancia evaluada")
    plt.ylabel("MSE")
    plt.legend(title="Modelo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/grafico_completo_parametros.png")
    plt.show()


# src/visualizador.py

import matplotlib.pyplot as plt
import numpy as np
import os

def graficar_coeficientes_modelo(modelo, feature_names, nombre_modelo):
    """
    Visualiza los coeficientes del modelo entrenado.
    Args:
        modelo: Instancia de modelo entrenado (ya ajustado).
        feature_names: Lista o array de nombres de las features.
        nombre_modelo: str, nombre del modelo (para guardar el gráfico).
    """
    coef = modelo.coef_
    plt.figure(figsize=(10, 6))
    colores = ['red' if c == 0 else 'steelblue' for c in coef]

    indices_ordenados = np.argsort(np.abs(coef))[::-1]
    plt.barh(np.array(feature_names)[indices_ordenados], coef[indices_ordenados], color=np.array(colores)[indices_ordenados])
    plt.xlabel("Coeficiente")
    plt.title(f"Importancia de Variables - {nombre_modelo.title()}")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/coeficientes_{nombre_modelo.lower()}.png"
    plt.savefig(path)
    plt.close()
    print(f"📊 Gráfico de coeficientes guardado en: {path}")
