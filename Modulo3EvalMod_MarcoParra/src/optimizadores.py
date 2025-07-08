import numpy as np
from src.excepciones import ParametrosNoConvergentesError

# ---------------------------
# 2. Funciones de costo y gradiente
# ---------------------------

def compute_mse(y_true, y_pred):
    """
    Calcula el Error Cuadrático Medio (MSE).

    Parámetros:
        y_true (array): Valores reales.
        y_pred (array): Valores predichos por el modelo.

    Retorna:
        float: MSE entre los valores reales y predichos.
    """
    return np.mean((y_true - y_pred) ** 2)


def compute_gradients(X, y, w, b):
    """
    Calcula el gradiente de la función de costo respecto a los parámetros w (pendiente) y b (sesgo).

    Parámetros:
        X (array): Valores de entrada.
        y (array): Valores reales.
        w (float): Valor actual de la pendiente.
        b (float): Valor actual del sesgo.

    Retorna:
        tuple: Derivadas parciales respecto a w y b.
    """
    n = len(X)
    y_pred = w * X + b
    dw = (-2 / n) * np.sum(X * (y - y_pred))  # derivada parcial respecto a w
    db = (-2 / n) * np.sum(y - y_pred)        # derivada parcial respecto a b
    return dw, db

# ---------------------------
# 3. Implementación de Descenso de Gradiente (GD)
# ---------------------------

def gradient_descent(X, y, lr=0.01, n_iter=200):
    """
    Ejecuta el algoritmo de descenso de gradiente sobre todos los datos (GD clásico).

    Parámetros:
        X (array): Entradas.
        y (array): Salidas reales.
        lr (float): Tasa de aprendizaje.
        n_iter (int): Número de iteraciones.

    Retorna:
        tuple: Parámetros finales (w, b), lista de costos, evolución de w y b.
    """
    w, b = 0.0, 0.0                     # Inicialización de parámetros
    costs, ws, bs = [], [], []         # Para registrar métricas y trayectorias

    for i in range(n_iter):
        dw, db = compute_gradients(X, y, w, b)  # Gradientes de la función de costo
        w -= lr * dw                            # Actualización del parámetro w
        b -= lr * db                            # Actualización del parámetro b

        # Cálculo e impresión del costo cada 10 iteraciones o en la última
        if i % 10 == 0 or i == n_iter - 1:
            cost = compute_mse(y, w * X + b)
            print(f"GD Iteración {i}: Costo = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")

        # Guardar historial
        costs.append(compute_mse(y, w * X + b))
        ws.append(w)
        bs.append(b)

    # Validación de convergencia: si el costo sigue siendo muy alto, lanzar excepción
    if costs[-1] > 1e3:
        raise ParametrosNoConvergentesError("GD no converge: costo final demasiado alto.")

    return w, b, costs, ws, bs

# ---------------------------
# 4. Implementación de Descenso de Gradiente Estocástico (SGD)
# ---------------------------

def stochastic_gradient_descent(X, y, lr=0.01, n_epochs=50):
    """
    Ejecuta el algoritmo de descenso de gradiente estocástico (SGD), actualizando con 1 muestra a la vez.

    Parámetros:
        X (array): Entradas.
        y (array): Salidas reales.
        lr (float): Tasa de aprendizaje.
        n_epochs (int): Número de épocas (pases por el dataset completo).

    Retorna:
        tuple: Parámetros finales (w, b), lista de costos, evolución de w y b.
    """
    w, b = 0.0, 0.0                     # Inicialización
    costs, ws, bs = [], [], []         # Historiales
    n = len(X)

    for epoch in range(n_epochs):
        for i in range(n):
            xi, yi = X[i], y[i]        # Punto individual
            y_pred = w * xi + b
            dw = -2 * xi * (yi - y_pred)
            db = -2 * (yi - y_pred)
            w -= lr * dw               # Actualización inmediata (SGD)
            b -= lr * db

        # Evaluar costo global tras cada época
        cost = compute_mse(y, w * X + b)
        print(f"SGD Época {epoch + 1}: Costo = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")

        # Guardar historial
        costs.append(cost)
        ws.append(w)
        bs.append(b)

    return w, b, costs, ws, bs
