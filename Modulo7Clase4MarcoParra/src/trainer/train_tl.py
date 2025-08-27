# src/trainer/train_tl.py
import os, json
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.utils.io import safe_save_json

def compile_model(
    model: tf.keras.Model,
    lr: float = 1e-3,
    loss: str = "sparse_categorical_crossentropy",
    metrics = ("accuracy",),
) -> tf.keras.Model:
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=list(metrics))
    return model

def train_model(
    model: tf.keras.Model,
    ds_train: tf.data.Dataset,
    ds_val: tf.data.Dataset,
    out_dir: str,
    run_name: str,
    epochs: int = 10,
    patience: int = 2,
) -> Tuple[tf.keras.callbacks.History, str]:
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, f"{run_name}_best.weights.h5")

    cbs = [
        ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True,
                        save_weights_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, patience-1), min_lr=1e-6, verbose=1),
    ]

    hist = model.fit(ds_train, validation_data=ds_val, epochs=epochs, callbacks=cbs, verbose=2)

    # history.json -> float nativos
    h = {k: [float(x) for x in v] for k, v in hist.history.items()}
    safe_save_json(h, os.path.join(out_dir, f"{run_name}_history.json"))
    return hist, ckpt

def evaluate_and_save(
    model: tf.keras.Model,
    ds_test: tf.data.Dataset,
    out_dir: str,
    run_name: str,
    idx_to_name: Dict[int,str]
) -> Dict:
    # predicciones
    y_true = []
    y_prob = []
    for x, y in ds_test:
        p = model.predict(x, verbose=0)
        y_prob.append(p)
        y_true.append(y.numpy())
    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = y_prob.argmax(axis=1)

    # métricas
    acc = float((y_pred == y_true).mean())

    # matriz de confusión
    num_classes = len(idx_to_name)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    # guardar
    test_report = {"accuracy": acc}
    safe_save_json(test_report, os.path.join(out_dir, f"{run_name}_test_report.json"))
    np.savetxt(os.path.join(out_dir, f"{run_name}_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    return {"accuracy": acc, "cm": cm, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
