import json
import os, json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_history_from_json(history_json_path: str, out_png_path: str):
    """
    history_json_path: path al archivo JSON con el historial de entrenamiento.
    out_png_path: path donde guardar la imagen PNG con las gráficas.
    """
    with open(history_json_path, "r") as f:
        hist = json.load(f)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # Loss
    ax[0].plot(hist.get("loss", []), label="train")
    ax[0].plot(hist.get("val_loss", []), label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    # Accuracy
    ax[1].plot(hist.get("accuracy", []), label="train")
    ax[1].plot(hist.get("val_accuracy", []), label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()

def plot_runs_comparison(runs, out_png_path: str):
    """
    runs: lista de dicts con keys: run, loss, opt, test_acc
    """
    import matplotlib.pyplot as plt
    labels = [r["run"] for r in runs]
    values = [r["test_acc"] for r in runs]
    plt.figure(figsize=(9, 4.5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=25, ha="right")
    plt.ylabel("Test accuracy"); plt.title("Comparación de corridas")
    plt.tight_layout(); plt.savefig(out_png_path, dpi=150); plt.close()


# ---------- Graficar directamente desde history ----------
def plot_history_inline(history):
    """Muestra curvas de loss/accuracy (train y val) desde el objeto History."""
    h = history.history
    acc  = h.get("accuracy", h.get("acc", []))
    val_acc = h.get("val_accuracy", h.get("val_acc", []))
    loss = h.get("loss", [])
    val_loss = h.get("val_loss", [])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(loss, label="train")
    if len(val_loss): ax[0].plot(val_loss, label="val")
    ax[0].set_title("Evolución de la pérdida")
    ax[0].set_xlabel("Época"); ax[0].legend()

    ax[1].plot(acc, label="train")
    if len(val_acc): ax[1].plot(val_acc, label="val")
    ax[1].set_title("Evolución de la accuracy")
    ax[1].set_xlabel("Época"); ax[1].legend()

    plt.tight_layout(); plt.show()


# ---------- QC del entrenamiento con history ----------
def _oscillation_score(series):
    """Heurística simple: cuántas veces aumenta de una época a otra (más = más oscilación)."""
    s = np.asarray(series, dtype=float)
    if s.size < 3: return 0
    return int(np.sum(s[1:] > s[:-1]))

def history_qc(history, test_acc: Optional[float] = None,
               expected_range: Tuple[float, float] = (0.95, 0.98)):
    """
    Genera un diagnóstico: mejora, gap train-val, oscilaciones y cumplimiento del rango esperado.
    expected_range por defecto es 95-98% (como en la pauta del curso).
    """
    h = history.history
    acc  = h.get("accuracy", [])
    val_acc = h.get("val_accuracy", [])
    loss = h.get("loss", [])
    val_loss = h.get("val_loss", [])

    final_train_acc = float(acc[-1]) if acc else None
    final_val_acc   = float(val_acc[-1]) if val_acc else None
    best_val_acc = float(max(val_acc)) if val_acc else None
    best_val_epoch = int(np.argmax(val_acc)) + 1 if val_acc else None
    gap = (final_train_acc - final_val_acc) if (final_train_acc is not None and final_val_acc is not None) else None

    loss_osc = _oscillation_score(val_loss) if len(val_loss) else None
    acc_improves = (len(acc) >= 2 and acc[-1] >= acc[0])

    expected_ok = None
    if test_acc is not None and expected_range is not None:
        lo, hi = expected_range
        expected_ok = (lo <= test_acc <= hi)

    notes = []
    if acc_improves: notes.append("✔ La accuracy de entrenamiento mejora a lo largo de las épocas.")
    else: notes.append("✘ La accuracy de entrenamiento no muestra mejora clara.")
    if gap is not None:
        if gap > 0.05: notes.append(f"⚠ Posible sobreajuste: gap train-val = {gap:.3f} (> 0.05).")
        else: notes.append(f"✔ Generalización razonable: gap train-val = {gap:.3f}.")
    if loss_osc is not None:
        if loss_osc > max(2, len(val_loss)//3):
            notes.append(f"⚠ Val_loss presenta oscilaciones frecuentes ({loss_osc} subidas).")
        else:
            notes.append(f"✔ Val_loss desciende de forma relativamente estable (subidas: {loss_osc}).")
    if expected_ok is not None:
        notes.append("✔ Accuracy de test en el rango esperado." if expected_ok
                     else "✘ Accuracy de test fuera del rango esperado.")

    return {
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "generalization_gap": gap,
        "val_loss_rises": loss_osc,
        "test_acc": test_acc,
        "expected_range": expected_range,
        "expected_ok": expected_ok,
        "notes": notes
    }

def save_history_qc_report(history, test_acc: Optional[float],
                           out_path: str,
                           expected_range: Tuple[float, float] = (0.95, 0.98)):   
    """Escribe un MD con el checklist de la pauta + números clave."""
    qc = history_qc(history, test_acc, expected_range)
    lines = []
    lines.append("# Reporte de entrenamiento (QC por history)\n")
    lines.append(f"- **Accuracy final (train)**: {qc['final_train_acc']:.4f}" if qc['final_train_acc'] is not None else "- Accuracy train: n/d")
    lines.append(f"- **Accuracy final (val)**: {qc['final_val_acc']:.4f}" if qc['final_val_acc'] is not None else "- Accuracy val: n/d")
    if qc["best_val_acc"] is not None:
        lines.append(f"- **Mejor val_accuracy**: {qc['best_val_acc']:.4f} (época {qc['best_val_epoch']})")
    if qc["generalization_gap"] is not None:
        lines.append(f"- **Gap train-val**: {qc['generalization_gap']:.4f}")
    if qc["val_loss_rises"] is not None:
        lines.append(f"- **Subidas de val_loss** (oscilación): {qc['val_loss_rises']}")
    if qc["test_acc"] is not None:
        lo, hi = qc["expected_range"]
        lines.append(f"- **Test accuracy**: {qc['test_acc']:.4f}  (rango esperado {lo:.2f}–{hi:.2f})")
        lines.append(f"- **Cumple rango esperado**: {'Sí' if qc['expected_ok'] else 'No'}")

    lines.append("\n## Observaciones")
    for n in qc["notes"]:
        lines.append(f"- {n}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return qc


def plot_runs_comparison(runs, out_png_path: str):
    """
    runs: lista de dicts con keys al menos: 'run', 'test_acc'.
    Genera una barra por corrida con la métrica de test.
    """
    import matplotlib.pyplot as plt
    labels = [r["run"] for r in runs]
    values = [r["test_acc"] for r in runs]
    plt.figure(figsize=(10, 4.5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=25, ha="right")
    plt.ylabel("Test accuracy")
    plt.title("Comparación de corridas")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()

