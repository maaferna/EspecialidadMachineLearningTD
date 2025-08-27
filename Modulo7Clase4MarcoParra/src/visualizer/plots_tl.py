# src/visualizer/plots_tl.py
import os, json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_history_from_json(history_json: str, out_png: str):
    with open(history_json, "r") as f:
        h = json.load(f)
    epochs = range(1, len(h["loss"]) + 1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(epochs, h["loss"], label="train"); plt.plot(epochs, h["val_loss"], label="val")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
    if "accuracy" in h:
        plt.subplot(1,2,2); plt.plot(epochs, h["accuracy"], label="train"); plt.plot(epochs, h["val_accuracy"], label="val")
        plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_png: str):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matriz de confusión"); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    # anotaciones
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=9)
    plt.xlabel("Predicción"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_predictions_grid(
    images, y_true, y_pred, idx_to_name: Dict[int,str], out_png: str, cols: int = 5, max_items: int = 25
):
    n = min(len(images), max_items)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(3*cols, 3*rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        x = images[i]
        if x.ndim == 3 and x.shape[-1] == 3:
            plt.imshow(np.clip(x / x.max(), 0, 1))
        else:
            plt.imshow(x.squeeze(), cmap="gray")
        t, p = int(y_true[i]), int(y_pred[i])
        title = f"T:{idx_to_name[t]}\nP:{idx_to_name[p]}"
        color = "green" if t == p else "red"
        plt.title(title, color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()
