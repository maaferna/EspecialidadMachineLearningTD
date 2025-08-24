# src/models/resnet_tabular.py
"""
Arquitectura ResNet ligera para datos tabulares (binaria).
- Bloques residuales con BatchNorm + ReLU.
- Opción de Dropout y regularización L2.
- Atajo (shortcut) con proyección cuando cambian las dimensiones.

Pensado para scoring crediticio / tabular en general.
"""

from typing import Optional

from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    ReLU,
)


def _l2(l2_coef: Optional[float]):
    """Devuelve un regularizador L2 (o None si no se especifica)."""
    return regularizers.l2(l2_coef) if l2_coef else None


def residual_block(x,
                   units: int,
                   dropout: float = 0.0,
                   l2: Optional[float] = None):
    """
    Bloque residual denso para tabulares.

    Estructura:
        x ----> [Dense -> BN -> ReLU -> Dropout? -> Dense -> BN] + shortcut -> ReLU -> Dropout?

    Notas:
    - Usamos proyección en el atajo cuando cambia la dimensionalidad
      (n_feats != units), para poder sumar tensores compatibles.
    - BN (Batch Normalization) ayuda a estabilizar entrenamientos con
      activaciones ReLU en tabulares.

    Parámetros
    ----------
    x : Tensor
        Entrada del bloque (KerasTensor).
    units : int
        Dimensionalidad interna del bloque (nº de neuronas Dense).
    dropout : float, opcional
        Probabilidad de Dropout después de cada activación principal.
    l2 : float, opcional
        Coeficiente de regularización L2 en las capas densas.

    Returns
    -------
    Tensor
        Salida del bloque residual (misma forma que `units`).
    """
    # Guardamos referencia para el atajo
    shortcut = x

    # Primera proyección lineal + normalización + activación
    out = Dense(units, activation=None, kernel_regularizer=_l2(l2))(x) 
    out = BatchNormalization()(out)
    out = ReLU()(out)

    # Dropout opcional (después de la activación)
    if dropout and dropout > 0.0:
        out = Dropout(dropout)(out)

    # Segunda proyección lineal + normalización (sin activación todavía)
    out = Dense(units, activation=None, kernel_regularizer=_l2(l2))(out)
    out = BatchNormalization()(out)

    # Si cambia la dimensionalidad, proyectamos el atajo para poder sumar
    in_dim = shortcut.shape[-1]
    if (in_dim is None) or (int(in_dim) != int(units)):
        shortcut = Dense(units, activation=None, kernel_regularizer=_l2(l2))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Suma residual y activación final
    out = Add()([shortcut, out])
    out = ReLU()(out)

    # Dropout opcional (al final del bloque)
    if dropout and dropout > 0.0:
        out = Dropout(dropout)(out)

    return out


def build_resnet_tabular(
    input_dim: int,
    stem_units: int = 128,
    blocks: int = 3,
    block_units: int = 128,
    dropout: float = 0.2,
    l2: Optional[float] = 1e-5,
) -> Model:
    """
    Construye un modelo ResNet para clasificación binaria en tabulares.

    Arquitectura (alto nivel):
        Input -> Dense(stem) -> BN -> ReLU -> Dropout?
                 -> [ residual_block x `blocks` ] -> Dense(1, sigmoid)

    Parámetros
    ----------
    input_dim : int
        Nº de características (columnas) tras el preprocesamiento.
    stem_units : int, opcional
        Nº de neuronas en el "stem" (capa inicial antes de los bloques).
    blocks : int, opcional
        Nº de bloques residuales a apilar.
    block_units : int, opcional
        Nº de neuronas internas en cada bloque residual.
    dropout : float, opcional
        Dropout a usar en el stem y en los bloques (0.0 desactiva).
    l2 : float, opcional
        Coeficiente de regularización L2 (None para desactivar).

    Returns
    -------
    Model
        Keras Model listo para compilar (salida con activación sigmoid).
    """
    # Entrada tabular densa (vector de longitud input_dim)
    inputs = Input(shape=(input_dim,), name="tabular_input")

    # "Stem": proyección inicial para llevar el espacio a `stem_units`
    x = Dense(stem_units, activation=None, kernel_regularizer=_l2(l2))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    if dropout and dropout > 0.0:
        x = Dropout(dropout)(x)

    # Bloques residuales apilados
    for _ in range(blocks):
        x = residual_block(x, units=block_units, dropout=dropout, l2=l2)

    # Capa de salida binaria para scoring (probabilidad de impago)
    outputs = Dense(1, activation="sigmoid", name="prob_default")(x)

    return Model(inputs=inputs, outputs=outputs, name="resnet_tabular")
