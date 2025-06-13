import time
import numpy as np
from numba import jit

# -----------------------------------------------
# ⏱️ Context manager personalizado para medir tiempo de ejecución
# -----------------------------------------------
class Timer:
    """
    Context manager para medir el tiempo de ejecución de un bloque de código.
    """
    def __init__(self, label):
        """
        Inicializa el temporizador con una etiqueta descriptiva.

        Args:
            label (str): Nombre que se mostrará junto con el tiempo registrado.
        """
        self.label = label

    def __enter__(self):
        """
        Guarda el tiempo de inicio del bloque medido.
        """
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Calcula el tiempo transcurrido al finalizar el bloque y lo imprime.
        """
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print(f"⏱️ {self.label}: {self.interval:.6f} segundos")


# -----------------------------------------------
# 🧮 Función intensiva utilizando bucles nativos de Python
# -----------------------------------------------
def suma_productos_bucle(arr1, arr2):
    """
    Calcula la suma de productos entre dos arreglos usando un bucle.

    Args:
        arr1 (list): Primer arreglo de números.
        arr2 (list): Segundo arreglo de números.

    Returns:
        int/float: Resultado de la suma de productos.
    """
    total = 0
    for i in range(len(arr1)):
        total += arr1[i] * arr2[i]
    return total


# -----------------------------------------------
# ⚡ Versión optimizada usando operaciones vectorizadas con NumPy
# -----------------------------------------------
def suma_productos_vectorizada(arr1, arr2):
    """
    Calcula la suma de productos usando operaciones vectorizadas con NumPy.

    Args:
        arr1 (ndarray): Primer arreglo NumPy.
        arr2 (ndarray): Segundo arreglo NumPy.

    Returns:
        float: Resultado de la suma de productos.
    """
    return np.sum(arr1 * arr2)


# -----------------------------------------------
# 🚀 Versión optimizada con Numba para acelerar bucles
# -----------------------------------------------
@jit(nopython=True)
def suma_productos_numba(arr1, arr2):
    """
    Calcula la suma de productos entre dos arreglos usando bucles,
    optimizados con compilación Just-In-Time mediante Numba.

    Args:
        arr1 (ndarray): Primer arreglo NumPy.
        arr2 (ndarray): Segundo arreglo NumPy.

    Returns:
        float: Resultado de la suma de productos.
    """
    total = 0
    for i in range(len(arr1)):
        total += arr1[i] * arr2[i]
    return total
