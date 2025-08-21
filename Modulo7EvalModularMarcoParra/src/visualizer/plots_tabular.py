# src/visualizer/plots_tabular.py
import os, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def plot_history_from_json(history_json: str, out_png: str):
    with open(history_json, "r") as f:
        h = json.load(f)
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()

def plot_confusion_matrix(cm, labels, out_png: str):
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusión")
    plt.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()

def plot_roc(roc_dict: Dict[str, Any], out_png: str):
    fpr = np.array(roc_dict["fpr"]); tpr = np.array(roc_dict["tpr"])
    fig, ax = plt.subplots(figsize=(4.5,4))
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0,1],[0,1],'--',color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
    ax.grid(True); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()
