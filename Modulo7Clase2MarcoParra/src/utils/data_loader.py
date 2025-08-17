# utils/data_loader.py
from tensorflow.keras.datasets import fashion_mnist

def load_raw_fashion_mnist():
    """Devuelve los datos crudos tal como vienen del dataset."""
    return fashion_mnist.load_data()

def get_class_names():
    return [
        "T-shirt/top","Trouser","Pullover","Dress","Coat",
        "Sandal","Shirt","Sneaker","Bag","Ankle boot"
    ]

