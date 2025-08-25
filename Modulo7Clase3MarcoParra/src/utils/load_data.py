# ── Utilidades: carga & preprocesamiento MNIST ─────────────────────────
"""
Utilidades de datos para MNIST (autoencoders).
- Carga dataset (tf.keras.datasets.mnist)
- Normaliza a [0,1]
- Aplana a vectores 784 para AEs densos (784→128→64→128→784)
"""

from typing import Tuple
import numpy as np
from tensorflow.keras.datasets import mnist


def load_mnist_flat(normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga MNIST, devuelve (x_train, x_test) como vectores de 784.
    No se utilizan etiquetas para autoencoders.
    """
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    if normalize:
        x_train /= 255.0
        x_test /= 255.0
    # aplanar 28*28 -> 784
    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))
    return x_train, x_test


def add_gaussian_noise(x: np.ndarray, sigma: float = 0.5, clip: bool = True) -> np.ndarray:
    """
    Agrega ruido gaussiano N(0, sigma) a imágenes (aplanadas).
    """
    noisy = x + np.random.normal(0.0, sigma, size=x.shape).astype("float32")
    if clip:
        noisy = np.clip(noisy, 0.0, 1.0)
    return noisy
