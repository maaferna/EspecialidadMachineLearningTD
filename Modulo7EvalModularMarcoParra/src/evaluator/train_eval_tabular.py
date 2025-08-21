# src/evaluator/train_eval_tabular.py
import os
import json
import numpy as np
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def compile_model(model, lr: float = 1e-3, loss: str = "binary_crossentropy"):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ---------- Serializadores seguros para JSON ----------

def _history_to_py(history_obj) -> Dict[str, Any]:
    """
    Convierte tf.keras.callbacks.History a un dict 100% serializable por json,
    casteando np.float32/np.float64 -> float y np.ndarray -> list(float).
    """
    py_hist: Dict[str, Any] = {}
    for key, values in history_obj.history.items():
        if isinstance(values, (list, tuple, np.ndarray)):
            py_hist[key] = [float(v) if hasattr(v, "__float__") else v for v in list(values)]
        else:
            py_hist[key] = float(values) if hasattr(values, "__float__") else values
    return py_hist


def _to_py(o):
    """Convierte recursivamente objetos NumPy a tipos nativos de Python."""
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.generic,)):  # np.int64, np.float32, etc.
        return o.item()
    return o


def save_json(d: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_py(d), f, indent=2)


# ---------- Entrenamiento ----------

def train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    out_dir: str,
    run_name: str,
    class_weight: Optional[Dict[int, float]] = None,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 5,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, f"{run_name}_best.weights.h5")
    callbacks = [
        ModelCheckpoint(
            ckpt,
            monitor="val_auc", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1
        ),
        EarlyStopping(
            monitor="val_auc", mode="max",
            patience=patience, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-5, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    # Guardar history serializable
    hist_path = os.path.join(out_dir, f"{run_name}_history.json")
    save_json(_history_to_py(history), hist_path)

    return history


def predict_proba(model, X):
    # Sigmoid -> probabilidades clase positiva
    return model.predict(X, verbose=0).ravel()
