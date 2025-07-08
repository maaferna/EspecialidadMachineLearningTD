import os
import numpy as np
import matplotlib.pyplot as plt

def generar_datos(n=100, semilla=42):
    """
    Genera datos sintéticos con relación lineal y ruido gaussiano.
    
    Parámetros:
    - n: número de muestras
    - semilla: para reproducibilidad

    Retorna:
    - x: variable independiente (n, 1)
    - y: variable dependiente (n, 1)
    """
    np.random.seed(semilla)
    x = np.random.rand(n, 1)
    y = 4 + 3 * x + np.random.randn(n, 1)  # y = 4 + 3x + ruido
    return x, y



def graficar_resultado(x, y, beta, nombre_archivo='grafico_resultado.png'):
    """
    Grafica los datos originales junto con la línea ajustada y guarda el gráfico.

    Parámetros:
    - x (ndarray): variable independiente
    - y (ndarray): variable dependiente
    - beta (ndarray): parámetros ajustados
    - nombre_archivo (str): nombre del archivo de salida dentro de /outputs
    """
    # Detecta el root del proyecto
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    ruta_salida = os.path.join(output_dir, nombre_archivo)

    # Generación del gráfico
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="blue", label="Datos originales")

    x_line = np.linspace(0, 1, 100).reshape(-1, 1)
    X_line = np.hstack([np.ones_like(x_line), x_line])
    y_pred = X_line @ beta

    plt.plot(x_line, y_pred, color="red", label="Línea ajustada")
    plt.title("Regresión Lineal (Álgebra Matricial)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ruta_salida)
    plt.show()

    print(f"✅ Gráfico guardado en: {ruta_salida}")


def ajustar_modelo(x, y):
    """
    Aplica álgebra matricial para calcular los parámetros del modelo lineal.

    Fórmula:
    β = (XᵀX)^-1 Xᵀy

    Parámetros:
    - x: variable independiente (n, 1)
    - y: variable dependiente (n, 1)

    Retorna:
    - beta: vector de parámetros ajustados (2, 1)
    """
    n = x.shape[0]
    X = np.hstack([np.ones((n, 1)), x])  # matriz diseño (n, 2)
    
    print(f"Shape de X: {X.shape}")
    print(f"Shape de Xᵀ: {X.T.shape}")
    print(f"Shape de XᵀX: {(X.T @ X).shape}")
    
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    print(f"Shape de β: {beta.shape}")
    return beta
