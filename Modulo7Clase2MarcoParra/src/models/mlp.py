# src/models/mlp_variants.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def build_mlp(input_shape=(28, 28),
              hidden_units=(128, 64),
              activations=("relu", "relu")):
    """
    Builder genérico de MLP:
      Flatten -> Dense(hidden_units[0], activations[0]) ->
      Dense(hidden_units[1], activations[1]) -> Dense(10, softmax)
    """
    assert len(hidden_units) == 2 and len(activations) == 2, "Se esperan dos capas ocultas"
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(hidden_units[0], activation=activations[0]),
        Dense(hidden_units[1], activation=activations[1]),
        Dense(10, activation="softmax")
    ])
    return model

def build_mlp_relu_relu(input_shape=(28, 28)):
    return build_mlp(input_shape, (128, 64), ("relu", "relu"))

def build_mlp_relu_tanh(input_shape=(28, 28)):
    # Requisito del enunciado: activaciones distintas (ReLU, Tanh)
    return build_mlp(input_shape, (128, 64), ("relu", "tanh"))
