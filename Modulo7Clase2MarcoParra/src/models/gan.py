# src/models/gan.py
from tensorflow.keras import layers, Model, Sequential

def build_generator(noise_dim: int = 100) -> Sequential:
    """
    Generador de imágenes MNIST (28x28x1) a partir de ruido.
    Utiliza capas Conv2DTranspose para upsampling.
    Salida: (batch_size, 28, 28, 1)
    Args:
        noise_dim: Dimensión del vector de ruido de entrada.
    Returns:
        Modelo Keras del generador.
    
    """
    model = Sequential(name="generator")
    model.add(layers.Input(shape=(noise_dim,)))
    model.add(layers.Dense(7*7*256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=1, padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization()); model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", use_bias=False, activation="tanh"))
    return model

def build_discriminator() -> Sequential:
    """
    Discriminador de imágenes MNIST (28x28x1) para clasificación real/falso.
    Utiliza capas Conv2D para downsampling.
    Salida: (batch_size, 1) con activación sigmoid.
    Returns:
        Modelo Keras del discriminador.
    """
    model = Sequential(name="discriminator")
    model.add(layers.Input(shape=(28,28,1)))
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2)); model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model
