"""
Autoencoders densos (flat) para MNIST (784).
- AE básico: 784 → ... → 784
- AE de-noising: misma arquitectura; el ruido se aplica en los datos de entrada
  (el builder no cambia; la diferencia está en cómo entrenas: x_noisy -> x_clean).

Se definen los builders con varios alias para compatibilidad con el main:
  - build_autoencoder_flat / build_autoencoder
  - build_denoising_autoencoder_flat / build_denoising_autoencoder / build_autoencoder_denoise_flat
"""

from typing import List, Optional
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, ReLU


def _l2(l2_coef: Optional[float]):
    return regularizers.l2(l2_coef) if l2_coef else None


def _mlp_encoder(x, hidden: List[int], dropout: float, l2: Optional[float]):
    for i, units in enumerate(hidden):
        x = Dense(units, activation=None, kernel_regularizer=_l2(l2), name=f"enc_dense_{i}")(x)
        x = BatchNormalization(name=f"enc_bn_{i}")(x)
        x = ReLU(name=f"enc_relu_{i}")(x)
        if dropout and dropout > 0.0:
            x = Dropout(dropout, name=f"enc_drop_{i}")(x)
    return x


def _mlp_decoder(x, hidden: List[int], dropout: float, l2: Optional[float]):
    # decodificador simétrico (ocultas en orden inverso)
    for j, units in enumerate(reversed(hidden)):
        x = Dense(units, activation=None, kernel_regularizer=_l2(l2), name=f"dec_dense_{j}")(x)
        x = BatchNormalization(name=f"dec_bn_{j}")(x)
        x = ReLU(name=f"dec_relu_{j}")(x)
        if dropout and dropout > 0.0:
            x = Dropout(dropout, name=f"dec_drop_{j}")(x)
    return x


def _build_autoencoder_flat_core(
    input_dim: int,
    hidden: List[int] = (512, 256, 128),
    latent_dim: int = 64,
    dropout: float = 0.1,
    l2: Optional[float] = 1e-6,
) -> Model:
    """
    Arquitectura MLP simétrica:
    input(784) -> [enc hidden...] -> latent(64) -> [dec hidden rev...] -> output(784, sigmoid)
    """
    inp = Input(shape=(input_dim,), name="ae_input")

    # Encoder
    x = _mlp_encoder(inp, list(hidden), dropout, l2)
    x = Dense(latent_dim, activation=None, kernel_regularizer=_l2(l2), name="latent_dense")(x)
    x = BatchNormalization(name="latent_bn")(x)
    x = ReLU(name="latent_relu")(x)

    # Decoder
    x = _mlp_decoder(x, list(hidden), dropout, l2)
    out = Dense(input_dim, activation="sigmoid", name="ae_output")(x)

    return Model(inp, out, name="ae_flat")


# ============== Builders públicos (con alias) ===================

def build_autoencoder_flat(
    input_dim: int,
    hidden: List[int] = (512, 256, 128),
    latent_dim: int = 64,
    dropout: float = 0.1,
    l2: Optional[float] = 1e-6,
) -> Model:
    """AE básico (flat)."""
    return _build_autoencoder_flat_core(
        input_dim=input_dim, hidden=list(hidden),
        latent_dim=latent_dim, dropout=dropout, l2=l2
    )


def build_denoising_autoencoder_flat(
    input_dim: int,
    hidden: List[int] = (512, 256, 128),
    latent_dim: int = 64,
    dropout: float = 0.1,
    l2: Optional[float] = 1e-6,
) -> Model:
    """
    AE de-noising (flat). Arquitectura igual al básico;
    la diferencia está en el *dataset* (entrada ruidosa -> salida limpia).
    """
    return _build_autoencoder_flat_core(
        input_dim=input_dim, hidden=list(hidden),
        latent_dim=latent_dim, dropout=dropout, l2=l2
    )


# ---- Aliases esperados por el main (compatibilidad) ----
build_autoencoder = build_autoencoder_flat

build_denoising_autoencoder = build_denoising_autoencoder_flat
build_autoencoder_denoise_flat = build_denoising_autoencoder_flat


__all__ = [
    "build_autoencoder_flat",
    "build_denoising_autoencoder_flat",
    "build_autoencoder",
    "build_denoising_autoencoder",
    "build_autoencoder_denoise_flat",
]
