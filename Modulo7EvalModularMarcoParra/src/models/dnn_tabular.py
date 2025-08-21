# src/models/dnn_tabular.py
from typing import List, Optional
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout, InputLayer, BatchNormalization

def build_dnn_tabular(
    input_dim: int,
    hidden_units: List[int] = [256, 128, 64],
    dropout: float = 0.2,
    l2: Optional[float] = 1e-5,
    bn: bool = True
):
    model = Sequential(name="dnn_tabular")
    model.add(InputLayer(input_shape=(input_dim,)))
    for u in hidden_units:
        model.add(Dense(u, activation="relu",
                        kernel_regularizer=regularizers.l2(l2) if l2 else None))
        if bn:
            model.add(BatchNormalization())
        if dropout and dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    return model
