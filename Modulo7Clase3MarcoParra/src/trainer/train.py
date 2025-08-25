# ── Entrenamiento y evaluación ────────────────────────────────────────
"""
Entrenamiento y evaluación para AEs (básico y denoising).
Guarda history.json (con floats Python), modelos y grids de reconstrucciones.
"""

# src/train.py
"""Entrenamiento y serialización robusta para autoencoders."""

import os
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.utils.utils import safe_save_json


def _to_serializable_history(hist: Dict) -> Dict:
    """Convierte floats/np.* a tipos nativos para JSON."""
    out = {}
    for k, v in hist.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            out[k] = [float(x) for x in v]
        else:
            out[k] = float(v) if isinstance(v, (np.floating,)) else v
    return out


def train_autoencoder(
    model: tf.keras.Model,
    x_in_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    out_dir: str,
    run_name: str,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 128,
    patience: int = 10,
) -> Tuple[tf.keras.callbacks.History, str]:

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])

    ckpt_path = os.path.join(out_dir, f"{run_name}_best.weights.h5")
    cbs = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss",
                        save_weights_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=max(1, patience-1), min_lr=1e-5, verbose=1),
    ]

    hist = model.fit(
        x=x_in_train, y=y_train,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size,
        shuffle=True, callbacks=cbs, verbose=2
    )

    # Guardar history (serializable)
    hist_path = os.path.join(out_dir, f"{run_name}_history.json")
    safe_save_json(_to_serializable_history(hist.history), hist_path)

    return hist, ckpt_path
