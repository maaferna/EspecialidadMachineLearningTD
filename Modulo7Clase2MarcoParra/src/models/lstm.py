# src/models/lstm.py
from typing import Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Bidirectional, Dropout, SpatialDropout1D
)
from tensorflow.keras import regularizers

def build_lstm_imdb(
    vocab_size: int = 30000,
    maxlen: int = 250,              # longitud de secuencia (padding)
    embed_dim: int = 128,           # dimensión de embeddings
    lstm_units: int = 64,           # capacidad por defecto
    bidirectional: bool = False,
    dropout: float = 0.5,           # dropout externo del LSTM
    rdropout: float = 0.3,          # dropout recurrente
    l2_coef: Optional[float] = 1e-5 # regularización L2 suave opcional
) -> Sequential:
    """
    LSTM para clasificación binaria (IMDb).
    - Masking de padding (mask_zero=True)
    - SpatialDropout1D sobre embeddings (reduce co-adaptaciones)
    - Dropout/recurrent_dropout en LSTM
    - L2 opcional
    Salida: Dense(1, sigmoid)
    """
    model = Sequential(name="lstm_imdb")

    # 🔑 Ignorar padding
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        input_length=maxlen,
        mask_zero=True
    ))

    # Regularización en la representación
    model.add(SpatialDropout1D(rate=0.3))

    # LSTM (opcionalmente bidireccional) con regularización
    lstm_layer = LSTM(
        lstm_units,
        dropout=dropout,
        recurrent_dropout=rdropout,
        kernel_regularizer=regularizers.l2(l2_coef) if l2_coef else None,
        recurrent_regularizer=regularizers.l2(l2_coef) if l2_coef else None,
    )
    rnn = Bidirectional(lstm_layer) if bidirectional else lstm_layer
    model.add(rnn)

    # Dropout final
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))
    return model
