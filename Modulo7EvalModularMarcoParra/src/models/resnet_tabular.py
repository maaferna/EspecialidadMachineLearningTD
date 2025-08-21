# src/models/resnet_tabular.py
from typing import Optional
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, ReLU, Add

def residual_block(x, units: int, dropout: float, l2: Optional[float]):
    shortcut = x
    out = Dense(units, kernel_regularizer=regularizers.l2(l2) if l2 else None)(x)
    out = BatchNormalization()(out); out = ReLU()(out)
    if dropout and dropout > 0: out = Dropout(dropout)(out)
    out = Dense(units, kernel_regularizer=regularizers.l2(l2) if l2 else None)(out)
    out = BatchNormalization()(out)
    # Proyección si cambia dimensión
    if shortcut.shape[-1] != out.shape[-1]:
        shortcut = Dense(units, kernel_regularizer=regularizers.l2(l2) if l2 else None)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    out = Add()([shortcut, out]); out = ReLU()(out)
    if dropout and dropout > 0: out = Dropout(dropout)(out)
    return out

def build_resnet_tabular(
    input_dim: int,
    stem_units: int = 128,
    blocks: int = 3,
    block_units: int = 128,
    dropout: float = 0.2,
    l2: Optional[float] = 1e-5,
):
    inp = Input((input_dim,), name="tabular_input")
    x = Dense(stem_units, activation=None,
              kernel_regularizer=regularizers.l2(l2) if l2 else None)(inp)
    x = BatchNormalization()(x); x = ReLU()(x)
    if dropout and dropout > 0: x = Dropout(dropout)(x)

    for _ in range(blocks):
        x = residual_block(x, units=block_units, dropout=dropout, l2=l2)

    out = Dense(1, activation="sigmoid")(x)
    return Model(inp, out, name="resnet_tabular")
