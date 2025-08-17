# src/experiments/grid.py
# ============================================================
# GRID para cumplir el enunciado (sin cargar datos aquí):
#  - Recibe datos ya cargados/normalizados/one-hot desde main
#  - Act: (relu, relu) y (relu, tanh)
#  - Loss: categorical_crossentropy, mse
#  - Opt: adam, sgd(momentum=0.9, lr=0.01)  [instancia nueva por run]
# Artefactos por run: *_history.json / *_curves.png / *_test_report.json / *_confusion_matrix.csv
# Resumen: experiments_summary.csv
# Comparación: experiments_comparison.png
# Reflexión: reflection_enunciado.md
# ============================================================

import os
import csv
import json
from typing import List, Tuple, Dict, Any

import tensorflow as tf

from src.evaluator.train_eval import compile_model, train_model, evaluate_on_test
from src.visualizer.plots import (
    plot_history_from_json,
    plot_runs_comparison,
    save_history_qc_report,
)

# Preferimos usar el builder modular si existe:
try:
    from src.models.mlp_variants import build_mlp  # build_mlp(activations=(act1, act2))
except Exception:
    # Fallback mínimo si no está mlp_variants.py
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    def build_mlp(input_shape=(28, 28), hidden_units=(128, 64), activations=("relu", "relu")):
        return Sequential([
            Flatten(input_shape=input_shape),
            Dense(hidden_units[0], activation=activations[0]),
            Dense(hidden_units[1], activation=activations[1]),
            Dense(10, activation="softmax"),
        ])


def _make_adam():
    """Devuelve SIEMPRE una instancia nueva de Adam (legacy si existe)."""
    try:
        from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
        return LegacyAdam()
    except Exception:
        return tf.keras.optimizers.Adam()


def _make_sgd():
    """Devuelve SIEMPRE una instancia nueva de SGD (legacy si existe)."""
    try:
        from tensorflow.keras.optimizers.legacy import SGD as LegacySGD
        return LegacySGD(learning_rate=0.01, momentum=0.9)
    except Exception:
        return tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)


def run_grid(
    data: Tuple[Any, Any, Any, Any],
    epochs: int = 10,
    batch_size: int = 32,
    validation_split: float = 0.2,
    out_dir: str = "outputs",
    expected_range: Tuple[float, float] = (0.88, 0.92),
) -> List[Dict[str, Any]]:
    """
    Ejecuta el grid completo sin cargar datos internamente.

    Parámetros
    ----------
    data : tuple
        (x_train, y_train, x_test, y_test) ya normalizados y con one-hot en y.
    epochs : int
    batch_size : int
    validation_split : float
    out_dir : str
    expected_range : (float, float)
        Rango esperado para val_acc/test_acc; se usa en el QC por run.

    Retorna
    -------
    List[Dict[str, Any]] : resumen de corridas.
    """
    os.makedirs(out_dir, exist_ok=True)

    x_train, y_train, x_test, y_test = data

    activations: List[Tuple[str, str]] = [("relu", "relu"), ("relu", "tanh")]
    losses = ["categorical_crossentropy", "mse"]
    optim_factories = {
        "adam": _make_adam,
        "sgd": _make_sgd,
    }

    results: List[Dict[str, Any]] = []

    for act1, act2 in activations:
        act_tag = f"{act1}_{act2}"
        for loss in losses:
            for opt_name, opt_fn in optim_factories.items():
                run_name = f"mlp_{act_tag}__{loss}__{opt_name}"
                print(f"\n=== RUN: {run_name} ===")

                # Asegura limpieza entre corridas (Keras 3)
                tf.keras.backend.clear_session()

                # Modelo y optimizador (instancia NUEVA por run)
                model = build_mlp(activations=(act1, act2))
                optimizer = opt_fn()

                # Compilación
                model = compile_model(model, loss=loss, optimizer=optimizer, metrics=["accuracy"])
                try:
                    if hasattr(model.optimizer, "build"):
                        model.optimizer.build(model.trainable_variables)
                except Exception:
                    pass

                # Entrenamiento (train_model guarda <run>_history.json)
                history = train_model(
                    model, x_train, y_train,
                    epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                    out_dir=out_dir, run_name=run_name
                )

                # Evaluación en test
                test_acc, _ = evaluate_on_test(
                    model, x_test, y_test, out_dir=out_dir, run_name=run_name
                )

                # Visualización / QC: SIEMPRE a través de src.visualizer.plots
                hist_json = os.path.join(out_dir, f"{run_name}_history.json")
                plot_history_from_json(
                    hist_json,
                    os.path.join(out_dir, f"{run_name}_curves.png")
                )
                save_history_qc_report(
                    history, test_acc,
                    out_path=os.path.join(out_dir, f"{run_name}_history_qc.md"),
                    expected_range=expected_range
                )

                # Resumen de métricas
                with open(hist_json, "r") as f:
                    hist = json.load(f)
                val_best_acc = float(max(hist.get("val_accuracy", [0.0])))

                results.append({
                    "run": run_name,
                    "act1": act1, "act2": act2,
                    "loss": loss, "opt": opt_name,
                    "epochs": epochs,
                    "val_best_acc": round(val_best_acc, 4),
                    "test_acc": round(float(test_acc), 4),
                })

    # Resumen CSV y comparación
    if results:
        csv_path = os.path.join(out_dir, "experiments_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader(); writer.writerows(results)

        plot_runs_comparison(results, os.path.join(out_dir, "experiments_comparison.png"))

        # Reflexión rápida (Markdown)
        best = max(results, key=lambda r: r["test_acc"])
        notes = [
            "Cross-entropy suele superar a MSE en clasificación multiclase (optimiza log-verosimilitud).",
            "Adam converge rápido en pocas épocas; SGD+momentum puede alcanzarlo con más épocas y ajuste de LR.",
            "ReLU en la primera capa favorece gradientes estables; Tanh en la segunda suaviza y puede ayudar a generalizar.",
            "Con más recursos: EarlyStopping(restore_best_weights), 12–15 épocas, Dropout(0.2), L2 suave y sweep de LR.",
        ]
        reflection_md = os.path.join(out_dir, "reflection_enunciado.md")
        with open(reflection_md, "w", encoding="utf-8") as f:
            f.write("# Reflexión sobre combinaciones (GRID)\n\n")
            f.write(f"- **Mejor combinación (test)**: `{best['loss']} + {best['opt']}` con **test_acc = {best['test_acc']:.4f}`.\n")
            f.write(f"- **Activaciones**: ({best['act1']}, {best['act2']}), mejor val_acc={best['val_best_acc']:.4f}.\n\n")
            f.write("## Observaciones\n")
            for n in notes: f.write(f"- {n}\n")
            f.write("\n## Tabla de corridas\n")
            for r in results:
                f.write(f"- {r['run']}: val_best_acc={r['val_best_acc']:.4f}, test_acc={r['test_acc']:.4f}\n")

    return results
