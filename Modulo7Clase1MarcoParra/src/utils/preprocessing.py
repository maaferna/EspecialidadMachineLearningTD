# utils/preprocessing.py
from tensorflow.keras.utils import to_categorical
from src.utils.data_loader import load_raw_fashion_mnist

def load_preprocess(dataset: str = "fashion",
                    one_hot: bool = True,
                    normalize: bool = True):
    """
    Carga y preprocesa el dataset Fashion-MNIST.

    Parámetros:
    - dataset: Nombre del dataset a cargar (actualmente solo "fashion").
    - one_hot: Si True, convierte las etiquetas a formato one-hot.
    - normalize: Si True, normaliza las imágenes a rango [0, 1].

    Retorna:
    - Tuple con los datos de entrenamiento y prueba: ((x_train, y_train), (x_test, y_test)).
    """
    (x_train, y_train), (x_test, y_test) = load_raw_fashion_mnist()

    if normalize:
        x_train = x_train.astype("float32") / 255.0
        x_test  = x_test.astype("float32") / 255.0

    if one_hot:
        y_train = to_categorical(y_train, 10)
        y_test  = to_categorical(y_test,  10)

    return (x_train, y_train), (x_test, y_test)
