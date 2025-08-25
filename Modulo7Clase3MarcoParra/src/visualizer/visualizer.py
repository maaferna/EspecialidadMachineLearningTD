# src/visualizer/visualizer.py
"""
Visualizaciones para autoencoders (MNIST flat).
- Curvas de entrenamiento desde history.json
- Grids de reconstrucción y de-noising

Firmas esperadas por scripts/main.py:
  - plot_history_from_json(path_json, out_png)
  - plot_reconstructions_grid(model, x, out_png, n=16)
  - plot_denoise_grid(model, x_noisy, x_clean, out_png, n=16)
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


# -------------------------- Utilidades internas -------------------------- #

def _ensure_array(x, name: str):
    """Asegura que x sea un array 2D o 4D compatible."""
    if hasattr(x, "numpy"):  # por si viene un tensor eager
        x = x.numpy()
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} debe ser np.ndarray; recibido: {type(x)}")
    return x


def _pred_batch(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Predice en lotes y retorna np.ndarray."""
    y = model.predict(x, batch_size=batch_size, verbose=0)
    if hasattr(y, "numpy"):
        y = y.numpy()
    return np.asarray(y)


def _to_grid(images: np.ndarray, n: int = 16) -> np.ndarray:
    """
    Convierte un batch de imágenes [N, 784] o [N, 28, 28] en un mosaico 4x4 (n=16).
    Escala a [0,1].
    """
    n = min(n, images.shape[0])
    imgs = images[:n]
    if imgs.ndim == 2:  # [N, 784]
        imgs = imgs.reshape((-1, 28, 28))
    elif imgs.ndim == 3:
        pass
    else:
        raise ValueError(f"Formas soportadas: [N,784] o [N,28,28]; recibido {imgs.shape}")

    imgs = np.clip(imgs, 0.0, 1.0)

    rows = cols = int(np.ceil(np.sqrt(n)))
    H = rows * 28
    W = cols * 28
    canvas = np.ones((H, W), dtype=np.float32)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = imgs[idx]
            idx += 1
    return canvas


def _save_gray(image_2d: np.ndarray, out_png: str, title: Optional[str] = None):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(image_2d, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# --------------------------- Funciones públicas -------------------------- #

def plot_history_from_json(path_json: str, out_png: str):
    """
    Lee history.json (Keras) y guarda un PNG con loss/val_loss y mae/val_mae si existen.
    """
    if not os.path.isfile(path_json):
        raise FileNotFoundError(f"history.json no encontrado: {path_json}")

    with open(path_json, "r") as f:
        hist = json.load(f)

    # Forzar listas a np.array por comodidad
    def arr(key): return np.asarray(hist.get(key, []), dtype=float)

    loss = arr("loss")
    val_loss = arr("val_loss")
    mae = arr("mae")
    val_mae = arr("val_mae")

    epochs = np.arange(1, max(len(loss), len(val_loss), len(mae), len(val_mae)) + 1)

    plt.figure(figsize=(10, 4))

    # Subplot 1: pérdidas
    plt.subplot(1, 2, 1)
    if loss.size:
        plt.plot(epochs, loss, label="loss")
    if val_loss.size:
        plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    plt.title("Curvas de pérdida")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: MAE
    plt.subplot(1, 2, 2)
    if mae.size:
        plt.plot(epochs, mae, label="mae")
    if val_mae.size:
        plt.plot(epochs, val_mae, label="val_mae")
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.title("Curvas de MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_reconstructions_grid(model, x: np.ndarray, out_png: str, n: int = 16):
    """
    Guarda un mosaico con pares (original, reconstrucción) en dos filas (o dos mosaicos apilados).
    - model: Keras Model (autoencoder)
    - x: np.ndarray de imágenes [N,784] o [N,28,28]
    """
    if hasattr(model, "predict") is False:
        raise TypeError("El primer argumento debe ser el modelo (Keras Model).")

    x = _ensure_array(x, "x")
    x = x[:n]
    if x.ndim == 2:
        x2d = x.reshape((-1, 28, 28))
    elif x.ndim == 3:
        x2d = x
        x = x.reshape((-1, 28 * 28))
    else:
        raise ValueError(f"Formas soportadas para x: [N,784] o [N,28,28]; recibido {x.shape}")

    x_pred = _pred_batch(model, x).reshape((-1, 28, 28))

    grid_orig = _to_grid(x2d, n=n)
    grid_reco = _to_grid(x_pred, n=n)
    grid = np.concatenate([grid_orig, grid_reco], axis=0)

    _save_gray(grid, out_png, title="Original (arriba) / Reconstrucción (abajo)")


def plot_denoise_grid(model, x_noisy: np.ndarray, x_clean: np.ndarray, out_png: str, n: int = 16):
    """
    Guarda un mosaico con (noisy, denoised, clean).
    - model: Keras Model (AE entrenado en de-noising)
    - x_noisy: entradas con ruido
    - x_clean: ground-truth limpio
    """
    if hasattr(model, "predict") is False:
        raise TypeError(
            "El primer argumento debe ser el modelo Keras. "
            "Firma esperada: plot_denoise_grid(model, x_noisy, x_clean, out_png, n=16)"
        )

    x_noisy = _ensure_array(x_noisy, "x_noisy")[:n]
    x_clean = _ensure_array(x_clean, "x_clean")[:n]

    # Asegurar formato [N,784]
    def to_flat(a):
        if a.ndim == 3:
            return a.reshape((-1, 28 * 28))
        elif a.ndim == 2:
            return a
        raise ValueError(f"Forma inválida (esperado [N,784] o [N,28,28]): {a.shape}")

    x_noisy_f = to_flat(x_noisy)
    x_clean_f = to_flat(x_clean)
    x_deno_f = _pred_batch(model, x_noisy_f)

    # A 2D
    x_noisy_2d = x_noisy_f.reshape((-1, 28, 28))
    x_deno_2d = x_deno_f.reshape((-1, 28, 28))
    x_clean_2d = x_clean_f.reshape((-1, 28, 28))

    grid_noisy = _to_grid(x_noisy_2d, n=n)
    grid_deno  = _to_grid(x_deno_2d,  n=n)
    grid_clean = _to_grid(x_clean_2d, n=n)

    grid = np.concatenate([grid_noisy, grid_deno, grid_clean], axis=0)
    _save_gray(grid, out_png, title="Noisy (arriba) / Denoised (medio) / Clean (abajo)")
