# src/models/dnn_tabular.py
"""
DNN para datos tabulares (binario) con paridad estructural respecto a ResNet tabular.
- Permite definir explícitamente `hidden_units` o construirlos en modo "paridad"
  a partir de (`stem_units`, `blocks`, `block_units`) para igualar profundidad/ancho
  con la ResNet y así comparar de forma justa.
- Incluye BatchNorm, Dropout y regularización L2 (mismos hiperparámetros que ResNet).

Ejemplo (paridad con ResNet: stem=128, blocks=3, block_units=128):
  hidden_units = [128] + [128] * (2 * 3)  -> 7 capas densas internas.
"""

from typing import List, Optional
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    InputLayer,
    ReLU,
)


def _l2(l2_coef: Optional[float]):
    """Devuelve regularizador L2 (o None)."""
    return regularizers.l2(l2_coef) if l2_coef else None


def _build_units_parity(stem_units: int, blocks: int, block_units: int) -> List[int]:
    """
    Construye una lista de unidades oculta para DNN con la misma profundidad “efectiva”
    que una ResNet tabular: 1 (stem) + 2*blocks (dos densas por bloque residual).
    """
    return [stem_units] + [block_units] * (2 * blocks)


def build_dnn_tabular(
    input_dim: int,
    hidden_units: Optional[List[int]] = None,
    *,
    # Parámetros para modo "paridad" (si hidden_units es None)
    stem_units: int = 128,
    blocks: int = 3,
    block_units: int = 128,
    # Regularización / normalización
    dropout: float = 0.2,
    l2: Optional[float] = 1e-5,
    bn: bool = True,
):
    """
    Construye una DNN para clasificación binaria en tabulares.

    Dos modos de uso:
    1) Especificar `hidden_units` explícitamente (lista de enteros).
    2) Dejar `hidden_units=None` y usar modo "paridad" con (`stem_units`, `blocks`,
       `block_units`) para igualar la profundidad/ancho a la ResNet tabular.

    Estructura de cada capa oculta:
        Dense (sin activación) -> BN? -> ReLU -> Dropout?
      (Se usa Dense sin activación para mantener simetría con la ResNet,
       donde la activación se aplica después del BatchNorm.)

    Parámetros
    ----------
    input_dim : int
        Nº de características (columnas) tras el preprocesamiento.
    hidden_units : list[int] o None
        Lista de neuronas por capa oculta. Si None, se infiere con paridad
        (1 + 2*blocks).
    stem_units : int
        Unidades del primer "bloque" (equivalente al stem de ResNet) en modo paridad.
    blocks : int
        Nº de bloques en ResNet (para construir 2*blocks capas adicionales) en modo paridad.
    block_units : int
        Unidades por capa oculta equivalente a cada sub-capa del bloque residual, en modo paridad.
    dropout : float
        Probabilidad de Dropout después de cada activación (0.0 desactiva).
    l2 : float o None
        Coeficiente de regularización L2 para las capas Dense (None desactiva).
    bn : bool
        Si True, incluye BatchNormalization tras cada Dense oculta.

    Returns
    -------
    Sequential
        Modelo Keras listo para compilar con salida sigmoid (probabilidad positiva).
    """
    # Si no se proveen unidades explícitas, construir en modo "paridad"
    if hidden_units is None:
        hidden_units = _build_units_parity(stem_units, blocks, block_units)

    model = Sequential(name="dnn_tabular")
    model.add(InputLayer(input_shape=(input_dim,)))

    for idx, units in enumerate(hidden_units):
        # Dense sin activación (simetría con ResNet: BN -> ReLU)
        model.add(Dense(units, activation=None, kernel_regularizer=_l2(l2)))

        if bn:
            model.add(BatchNormalization())

        # Activación explícita para mantener el orden BN->ReLU
        model.add(ReLU(name=f"relu_{idx}"))

        if dropout and dropout > 0.0:
            model.add(Dropout(dropout))

    # Capa de salida binaria
    model.add(Dense(1, activation="sigmoid", name="prob_default"))
    return model
