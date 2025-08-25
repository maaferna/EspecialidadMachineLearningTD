# scripts/main.py
# ============================================================
# Orquestador de Autoencoders (MNIST, flat 784) - Modular
#   - basic   : AE denso 784→...→784
#   - denoise : AE denso con ruido gaussiano en la entrada
#
# Este main:
#   * Importa módulos del paquete src.*
#   * Llama funciones concretas (sin reflexión frágil)
#   * Guarda history.json (serializable), pesos y gráficos
#   * No implementa lógica de entrenamiento/plots aquí (solo orquesta)
# ============================================================

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

# ─────────────────────────────── PYTHONPATH ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────── Imports SRC ──────────────────────────────
import src.utils.load_data as data
import src.models.autoenconders as ae
import src.trainer.train as trainer
import src.visualizer.visualizer as viz


# ─────────────────────────── Utilidades GPU (opcional) ────────────────────
def enable_memory_growth():
    """Activa memory growth en las GPUs disponibles para evitar OOM inicial."""
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    return gpus


# ─────────────────────────────── Construcción AE ──────────────────────────
def build_basic_autoencoder(input_dim: int) -> tf.keras.Model:
    """
    Construye AE denso (flat). Se admiten nombres alternativos en tu módulo:
      - build_autoencoder_flat
      - build_autoencoder
    """
    if hasattr(ae, "build_autoencoder_flat"):
        return ae.build_autoencoder_flat(input_dim=input_dim)
    if hasattr(ae, "build_autoencoder"):
        return ae.build_autoencoder(input_dim=input_dim)
    raise AttributeError(
        "No encontré un builder básico en src.models.autoenconders. "
        "Define 'build_autoencoder_flat(input_dim)' o 'build_autoencoder(input_dim)'."
    )


def build_denoise_autoencoder(input_dim: int) -> tf.keras.Model:
    """
    Construye AE denso de-noising. Nombres alternativos aceptados:
      - build_denoising_autoencoder_flat
      - build_denoising_autoencoder
      - build_autoencoder_denoise_flat
    """
    for name in (
        "build_denoising_autoencoder_flat",
        "build_denoising_autoencoder",
        "build_autoencoder_denoise_flat",
    ):
        if hasattr(ae, name):
            return getattr(ae, name)(input_dim=input_dim)
    raise AttributeError(
        "No encontré un builder de-noising en src.models.autoenconders. "
        "Define 'build_denoising_autoencoder_flat(input_dim)' (o alguno de los alias)."
    )


# ─────────────────────────────── Ejecutor ─────────────────────────────────
def run_basic(
    out_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
):
    # 1) Datos (x_train, x_test) ∈ [0,1], shape: (N, 784)
    x_train, x_test = data.load_mnist_flat(normalize=True)
    input_dim = x_train.shape[1]
    print(f"Loaded MNIST flat: train={x_train.shape}, test={x_test.shape}")

    # 2) Modelo básico
    model = build_basic_autoencoder(input_dim=input_dim)

    # 3) Entrenamiento (y = x)
    run_name = f"ae_basic__lr{lr}__bs{batch_size}__ep{epochs}"
    hist, ckpt_path = trainer.train_autoencoder(
        model=model,
        x_in_train=x_train, x_val=x_test,  # validamos en test para simplicidad
        y_train=x_train, y_val=x_test,
        out_dir=out_dir, run_name=run_name,
        lr=lr, epochs=epochs, batch_size=batch_size, patience=patience
    )

    # 4) Cargar mejores pesos
    if os.path.isfile(ckpt_path):
        model.load_weights(ckpt_path)

    # 5) Curvas y reconstrucciones
    hist_json = os.path.join(out_dir, f"{run_name}_history.json")
    curves_png = os.path.join(out_dir, f"{run_name}_curves.png")
    if hasattr(viz, "plot_history_from_json"):
        viz.plot_history_from_json(hist_json, curves_png)
    else:
        print("⚠ No se encontró viz.plot_history_from_json; omito curvas.")

    recons_png = os.path.join(out_dir, f"{run_name}_recons.png")
    if hasattr(viz, "plot_reconstructions_grid"):
        # muestra originales vs reconstruidas en una grilla
        viz.plot_reconstructions_grid(model, x_test[:16], out_png=recons_png)
    else:
        print("⚠ No se encontró viz.plot_reconstructions_grid; omito grilla.")


def run_denoise(
    out_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    sigma: float,
):
    # 1) Datos (x_train, x_test) ∈ [0,1], shape: (N, 784)
    x_train, x_test = data.load_mnist_flat(normalize=True)
    input_dim = x_train.shape[1]
    print(f"Loaded MNIST flat: train={x_train.shape}, test={x_test.shape}")

    # 2) Ruido gaussiano en la entrada (manteniendo target limpio)
    x_train_noisy = data.add_gaussian_noise(x_train, sigma=sigma, clip=True)
    x_test_noisy = data.add_gaussian_noise(x_test, sigma=sigma, clip=True)

    # 3) Modelo de-noising
    model = build_denoise_autoencoder(input_dim=input_dim)

    # 4) Entrenamiento (entrada ruidosa → salida limpia)
    run_name = f"ae_denoise__lr{lr}__bs{batch_size}__ep{epochs}__sigma{sigma}"
    hist, ckpt_path = trainer.train_autoencoder(
        model=model,
        x_in_train=x_train_noisy, x_val=x_test_noisy,
        y_train=x_train,      y_val=x_test,
        out_dir=out_dir, run_name=run_name,
        lr=lr, epochs=epochs, batch_size=batch_size, patience=patience
    )

    # 5) Cargar mejores pesos
    if os.path.isfile(ckpt_path):
        model.load_weights(ckpt_path)

    # 6) Curvas y grilla de denoise
    hist_json = os.path.join(out_dir, f"{run_name}_history.json")
    curves_png = os.path.join(out_dir, f"{run_name}_curves.png")
    if hasattr(viz, "plot_history_from_json"):
        viz.plot_history_from_json(hist_json, curves_png)
    else:
        print("⚠ No se encontró viz.plot_history_from_json; omito curvas.")

    denoise_png = os.path.join(out_dir, f"{run_name}_denoise_recons.png")
    if hasattr(viz, "plot_denoise_grid"):
        viz.plot_denoise_grid(model, x_test_noisy[:16], x_test[:16], out_png=denoise_png)
    else:
        print("⚠ No se encontró viz.plot_denoise_grid; omito grilla de denoise.")


# ─────────────────────────────── CLI / Main ────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Autoencoders (MNIST flat) — basic/denoise")
    p.add_argument("--mode", choices=["basic", "denoise"], default="basic")
    p.add_argument("--out-dir", default="outputs_ae")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.5, help="std del ruido gaussiano (denoise)")
    p.add_argument("--gpu-visible-devices", default=None,
                   help="Ej: '0' o '1' o '0,1'. Si None, usa todas las GPUs visibles.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # GPUs visibles (opcional)
    if args.gpu_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_visible_devices)

    gpus = enable_memory_growth()
    print("GPUs visibles:", [g.name for g in gpus] if gpus else "[]")

    # Ejecutar modo seleccionado
    if args.mode == "basic":
        run_basic(
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )
    else:
        run_denoise(
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            sigma=args.sigma,
        )


if __name__ == "__main__":
    main()
