"""
Runner de Credit Scoring (German Credit UCI) con DNN y ResNet en paridad estructural.
- Carga y EDA básica.
- Preprocesamiento + splits.
- Entrenamiento y evaluación (ROC, F1, accuracy, matriz de confusión).
- Costeo de errores (FP/FN).
- SHAP/LIME opcionales para explicabilidad.
- Paridad de arquitectura entre DNN y ResNet para comparación justa.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# --- PYTHONPATH: raíz del proyecto ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports del proyecto ---
from src.utils.credit_data import (
    eda_basic,
    load_csv,
    load_uci_german,
    prepare_splits,
)
from src.models.dnn_tabular import build_dnn_tabular
from src.models.resnet_tabular import build_resnet_tabular
from src.evaluator.train_eval_tabular import compile_model, train_model, save_json
from src.evaluator.metrics_tabular import evaluate_classification, cost_analysis
from src.visualizer.plots_tabular import (
    plot_confusion_matrix,
    plot_history_from_json,
    plot_roc,
)
from src.explain.shap_lime import (
    shap_summary_kernelexplainer,
    lime_explain_instances,
)


# ============================
# Configuración global paridad
# ============================
PARITY_STEM_UNITS = 256
PARITY_BLOCKS = 3
PARITY_BLOCK_UNITS = 256
PARITY_DROPOUT = 0.2
PARITY_L2 = 1e-5


def enable_memory_growth():
    """Activa growth dinámico de memoria en todas las GPUs visibles."""
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    return gpus


def main():
    """Punto de entrada del experimento de scoring crediticio."""
    ap = argparse.ArgumentParser(
        description="Credit Scoring con DNN/ResNet + SHAP/LIME (German Credit UCI)"
    )
    ap.add_argument("--data", default="uci", help="Ruta CSV o 'uci' para descargar desde UCI")
    ap.add_argument("--model", choices=["dnn", "resnet", "both"], default="both")
    ap.add_argument("--out-dir", default="outputs_credit")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--mixed",
        action="store_true",
        help="Activa mixed precision (recomendado en RTX 40xx).",
    )
    ap.add_argument("--cost-fp", type=float, default=1.0, help="Costo FP (otorgar a quien no paga)")
    ap.add_argument("--cost-fn", type=float, default=5.0, help="Costo FN (rechazar a quien sí paga)")
    args = ap.parse_args()

    # --- Semillas / salida ---
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # --- GPU y mixed precision (opcional) ---
    gpus = enable_memory_growth()
    print("GPUs visibles:", gpus)
    if args.mixed:
        try:
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision ON")
        except Exception as e:
            print("No se pudo activar mixed precision:", e)

    # ======================
    # 1) Carga + EDA básica
    # ======================
    if args.data.lower() == "uci":
        df, target = load_uci_german(cache_dir=".cache_uci")  # target = 'class'
    else:
        df, target = load_csv(args.data, target="class")  # target fijo por requerimiento

    info = eda_basic(df, target)
    print("EDA:", json.dumps(info, indent=2))
    with open(os.path.join(args.out_dir, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # ====================================
    # 2) Splits + preprocesamiento tabular
    # ====================================
    (X_train, y_train), (X_val, y_val), (X_test, y_test), ct, meta = prepare_splits(df, target)
    n_in = meta["n_features_out"]
    class_weight = meta["class_weight"]

    # Nombres de features (para SHAP/LIME)
    try:
        num_names = meta["num_cols"]
        cat_encoder = ct.named_transformers_["cat"]
        cat_input = cat_encoder.get_feature_names_out(meta["cat_cols"]).tolist()
        feature_names = num_names + cat_input
    except Exception:
        feature_names = [f"f{i}" for i in range(n_in)]

    # ===========================
    # 3) Definir modelos a correr
    # ===========================
    # Modo paridad: DNN y ResNet con la MISMA capacidad efectiva.
    candidates = []
    if args.model in ["dnn", "both"]:
        # DNN en modo paridad: 1 (stem) + 2*blocks capas internas, todas de 256
        candidates.append(
            (
                "dnn",
                lambda: build_dnn_tabular(
                    input_dim=n_in,
                    hidden_units=None,  # <- activa paridad
                    stem_units=PARITY_STEM_UNITS,
                    blocks=PARITY_BLOCKS,
                    block_units=PARITY_BLOCK_UNITS,
                    dropout=PARITY_DROPOUT,
                    l2=PARITY_L2,
                    bn=True,
                ),
            )
        )
    if args.model in ["resnet", "both"]:
        candidates.append(
            (
                "resnet",
                lambda: build_resnet_tabular(
                    input_dim=n_in,
                    stem_units=PARITY_STEM_UNITS,
                    blocks=PARITY_BLOCKS,
                    block_units=PARITY_BLOCK_UNITS,
                    dropout=PARITY_DROPOUT,
                    l2=PARITY_L2,
                ),
            )
        )

    # ==============================
    # 4) Entrenamiento + evaluación
    # ==============================
    runs = []
    for name, builder in candidates:
        run = f"{name}__lr{args.lr}__bs{args.batch_size}__ep{args.epochs}"
        print(f"\n=== Entrenando {run} ===")

        # Construir y compilar
        model = builder()
        model = compile_model(model, lr=args.lr, loss="binary_crossentropy")

        # Entrenar (train_model guarda history.json)
        hist = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            out_dir=args.out_dir,
            run_name=run,
            class_weight=class_weight,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=5,
        )

        # Curvas desde el history.json
        plot_history_from_json(
            os.path.join(args.out_dir, f"{run}_history.json"),
            os.path.join(args.out_dir, f"{run}_curves.png"),
        )

        # Evaluación final en test
        y_proba = model.predict(X_test, verbose=0).ravel()
        metrics = evaluate_classification(y_test, y_proba, threshold=0.5)
        cm = np.array(metrics["confusion_matrix"])
        costs = cost_analysis(cm, cost_fp=args.cost_fp, cost_fn=args.cost_fn)

        # Guardado (json seguro con casting dentro de save_json)
        save_json(metrics, os.path.join(args.out_dir, f"{run}_test_report.json"))
        save_json(costs, os.path.join(args.out_dir, f"{run}_costs.json"))
        np.savetxt(
            os.path.join(args.out_dir, f"{run}_confusion_matrix.csv"),
            cm,
            fmt="%d",
            delimiter=",",
        )

        # Plots derivados
        plot_confusion_matrix(
            cm,
            labels=["No-Impago(0)", "Impago(1)"],
            out_png=os.path.join(args.out_dir, f"{run}_cm.png"),
        )
        plot_roc(metrics["roc_curve"], out_png=os.path.join(args.out_dir, f"{run}_roc.png"))

        runs.append(
            {
                "run": run,
                "roc_auc": float(metrics["roc_auc"]),
                "f1": float(metrics["f1"]),
                "accuracy": float(metrics["accuracy"]),
            }
        )

        # ==================
        # 5) Explicabilidad
        # ==================
        # SHAP KernelExplainer (muestreo para tiempo razonable)
        try:
            n_bg = min(200, len(X_train))
            n_x = min(200, len(X_test))
            shap_summary_kernelexplainer(
                model,
                X_train[:n_bg],
                X_test[:n_x],
                feature_names,
                out_png=os.path.join(args.out_dir, f"{run}_shap_summary.png"),
            )
        except Exception as e:
            print("SHAP falló (continuo):", e)

        # LIME (3 instancias) – adaptando a predict_proba con salida [n, 2]
        try:
            def predict_proba(x):
                p = model.predict(x, verbose=0).ravel()
                return np.vstack([1.0 - p, p]).T

            lime_explain_instances(
                type("M", (), {"predict_proba": predict_proba}),
                X_train,
                X_test,
                feature_names,
                class_names=["No-Impago", "Impago"],
                out_dir=os.path.join(args.out_dir, f"{run}_lime"),
                num_instances=3,
            )
        except Exception as e:
            print("LIME falló (continuo):", e)

    # ========================
    # 6) Resumen y selección
    # ========================
    with open(os.path.join(args.out_dir, "summary_runs.json"), "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    best = max(runs, key=lambda r: r["roc_auc"])
    print("\n✅ Mejor modelo:", best)


if __name__ == "__main__":
    main()
