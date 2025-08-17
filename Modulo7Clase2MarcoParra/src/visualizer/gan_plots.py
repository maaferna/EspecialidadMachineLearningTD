# src/visualizer/gan_plots.py
import os
import numpy as np
import matplotlib.pyplot as plt

def save_image_grid(images: np.ndarray, out_path: str, nrows: int = 5, ncols: int = 5):
    """
    Guarda una grilla nrows x ncols de imágenes (28x28, escala [-1,1] o [0,1]).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imgs = images.copy()
    # Reescala de [-1,1] a [0,1] si corresponde
    if imgs.min() < 0:
        imgs = (imgs + 1.0) / 2.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.imshow(imgs[idx].squeeze(), cmap="gray")
            ax.axis("off")
            idx += 1
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def plot_gan_losses(history: dict, out_path: str):
    """
    history: dict con listas 'd_loss', 'g_loss'
    """
    plt.figure(figsize=(7,4))
    plt.plot(history.get("d_loss", []), label="Discriminator")
    plt.plot(history.get("g_loss", []), label="Generator")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("GAN training losses")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
