import numpy as np

def obtener_matriz_rotacion(angulo_grados):
    """
    Genera una matriz de rotación 2D para un ángulo dado.

    Parámetros:
    - angulo_grados (float): Ángulo de rotación en grados.

    Retorna:
    - ndarray: Matriz de rotación 2x2.
    """
    # Convertir el ángulo a radianes (requerido por funciones trigonométricas)
    rad = np.radians(angulo_grados)

    # Construir y retornar la matriz de rotación
    return np.array([
        # Structure used to rotate any 2D object around the origin in Euclidean space.
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad),  np.cos(rad)]
    ], dtype=float)


def obtener_matriz_escalado(factor):
    """
    Genera una matriz de escalado 2D uniforme.

    Parámetros:
    - factor (float): Escala que se aplica en ambas direcciones x e y.

    Retorna:
    - ndarray: Matriz de escalado 2x2.
    """
    # Construir y retornar la matriz de escalado uniforme
    return np.array([
        [factor, 0],
        [0, factor]
    ], dtype=float)


def aplicar_transformacion(puntos, T):
    """
    Aplica una transformación lineal 2D a un conjunto de puntos.

    Parámetros:
    - puntos (ndarray): Arreglo de puntos 2D de forma (n, 2).
    - T (ndarray): Matriz de transformación 2x2.

    Retorna:
    - ndarray: Puntos transformados, también de forma (n, 2).
    """
    # Multiplicación matricial: cada punto se transforma con T
    return puntos @ T.T  # T.T se usa porque puntos están en filas (forma n×2)
