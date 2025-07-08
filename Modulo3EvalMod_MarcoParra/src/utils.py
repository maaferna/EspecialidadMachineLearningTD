"""
Este script implementa y compara diferentes algoritmos de optimización (GD, SGD, opcionalmente Adam)
para resolver un problema de regresión lineal utilizando MSE como función de costo.

Estructura pensada para reproducibilidad, claridad y evaluación gráfica.
"""

import numpy as np
from src.excepciones import DatosInsuficientesError

# ---------------------------
# 1. Generar datos sintéticos reproducibles
# ---------------------------
def generar_datos_sinteticos(n=100, intervalo=(0, 10), coeficiente=3, intercepto=4, ruido_std=1, seed=42):
    """
    Genera datos sintéticos para un problema de regresión lineal.

    Parámetros:
    - n: número de muestras
    - intervalo: rango de valores de x
    - coeficiente: pendiente verdadera
    - intercepto: sesgo verdadero
    - ruido_std: desviación estándar del ruido
    - seed: semilla para reproducibilidad

    Retorna:
    - x: valores de entrada
    - y: valores de salida con ruido
    """
    if seed is not None:
        np.random.seed(seed)

    if n < 2:
        raise DatosInsuficientesError("Se requieren al menos 2 datos para entrenamiento.")
    
    # Genera n valores de la variable independiente x, distribuidos uniformemente 
    # en el intervalo especificado (por defecto entre 0 y 10).
    x = np.random.uniform(intervalo[0], intervalo[1], n)
    # Genera el ruido gaussiano con media 0 y desviación estándar especificada.
    ruido = np.random.normal(0, ruido_std, n)
    # Calcula los valores de la variable dependiente y usando la ecuación de la recta
    # y = mx + b, donde m es el coeficiente (pendiente) y b es el intercepto (sesgo).
    y = intercepto + coeficiente * x + ruido
    return x, y



