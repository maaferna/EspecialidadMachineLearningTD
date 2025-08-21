# scripts/credit_main.py
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.credit_data import load_csv, load_uci_german, eda_basic, prepare_splits
from src.models.dnn_tabular import build_dnn_tabular
from src.models.resnet_tabular import build_resnet_tabular
from src.evaluator.train_eval_tabular import compile_model, train_model, save_json
from src.evaluator.metrics_tabular import evaluate_classification, cost_analysis
from src.visualizer.plots_tabular import plot_history_from_json, plot_confusion_matrix, plot_roc
from src.explain.shap_lime import shap_summary_kernelexplainer, lime_explain_instances
import numpy as np

def enable_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass
    return gpus

def main():
    ap = argparse.ArgumentParser(description="Credit Scoring con DNN/ResNet + SHAP/LIME (German Credit UCI)")
    ap.add_argument("--data", default="uci", help="Ruta CSV o 'uci' para descargar desde UCI")
    ap.add_argument("--model", choices=["dnn","resnet","both"], default="both")
    ap.add_argument("--out-dir", default="outputs_credit")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed", action="store_true", help="Mixed precision")
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--cost-fn", type=float, default=5.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed); tf.random.set_seed(args.seed)

    gpus = enable_memory_growth()
    print("GPUs visibles:", gpus)
    if args.mixed:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision ON")
        except Exception as e:
            print("No se pudo activar mixed precision:", e)

    # 1) Carga (UCI por defecto) y EDA
    if args.data.lower() == "uci":
        df, target = load_uci_german(cache_dir=".cache_uci")
        # target es SIEMPRE 'class' en este flujo
    else:
        df, target = load_csv(args.data, target="class")  # target fijo 'class' por requerimiento

    info = eda_basic(df, target)
    print("EDA:", json.dumps(info, indent=2))
    with open(os.path.join(args.out_dir, "eda_summary.json"), "w") as f:
        json.dump(info, f, indent=2)

    # 2) Splits + preprocesamiento
    (X_train, y_train), (X_val, y_val), (X_test, y_test), ct, meta = prepare_splits(df, target)
    n_in = meta["n_features_out"]
    class_weight = meta["class_weight"]

    # nombres de features para SHAP/LIME
    try:
        num_names = meta["num_cols"]
        cat_encoder = ct.named_transformers_["cat"]
        cat_input = cat_encoder.get_feature_names_out(meta["cat_cols"]).tolist()
        feature_names = num_names + cat_input
    except Exception:
        feature_names = [f"f{i}" for i in range(n_in)]

    candidates = []
    if args.model in ["dnn","both"]:
        candidates.append(("dnn", lambda: build_dnn_tabular(n_in, [256,128,64], 0.2, 1e-5, True)))
    if args.model in ["resnet","both"]:
        candidates.append(("resnet", lambda: build_resnet_tabular(n_in, stem_units=128, blocks=3, block_units=128, dropout=0.2, l2=1e-5)))

    runs = []
    for name, builder in candidates:
        run = f"{name}__lr{args.lr}__bs{args.batch_size}__ep{args.epochs}"
        print(f"\n=== Entrenando {run} ===")
        model = builder()
        model = compile_model(model, lr=args.lr, loss="binary_crossentropy")

        hist = train_model(
            model,
            X_train, y_train,
            X_val, y_val,
            out_dir=args.out_dir,
            run_name=run,
            class_weight=class_weight,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=5
        )

        # Curvas
        plot_history_from_json(
            os.path.join(args.out_dir, f"{run}_history.json"),
            os.path.join(args.out_dir, f"{run}_curves.png")
        )

        # Evaluación
        y_proba = model.predict(X_test, verbose=0).ravel()
        metrics = evaluate_classification(y_test, y_proba, threshold=0.5)
        cm = np.array(metrics["confusion_matrix"])
        costs = cost_analysis(cm, cost_fp=args.cost_fp, cost_fn=args.cost_fn)

        # Guardar
        save_json(metrics, os.path.join(args.out_dir, f"{run}_test_report.json"))
        save_json(costs,   os.path.join(args.out_dir, f"{run}_costs.json"))
        np.savetxt(os.path.join(args.out_dir, f"{run}_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

        # Plots
        plot_confusion_matrix(cm, labels=["No-Impago(0)","Impago(1)"],
                              out_png=os.path.join(args.out_dir, f"{run}_cm.png"))
        plot_roc(metrics["roc_curve"], out_png=os.path.join(args.out_dir, f"{run}_roc.png"))

        runs.append({"run": run, "roc_auc": metrics["roc_auc"], "f1": metrics["f1"], "accuracy": metrics["accuracy"]})

        # Explicabilidad (muestreo para tiempo razonable)
        try:
            n_bg = min(200, len(X_train))
            n_x  = min(200, len(X_test))
            shap_summary_kernelexplainer(
                model, X_train[:n_bg], X_test[:n_x], feature_names,
                out_png=os.path.join(args.out_dir, f"{run}_shap_summary.png")
            )
        except Exception as e:
            print("SHAP falló (continuo):", e)

        try:
            # LIME (3 instancias): adaptar predict_proba a forma [n,2]
            def predict_proba(X):
                p = model.predict(X, verbose=0).ravel()
                return np.vstack([1-p, p]).T
            lime_explain_instances(
                type("M", (), {"predict_proba": predict_proba}),
                X_train, X_test, feature_names,
                class_names=["No-Impago","Impago"],
                out_dir=os.path.join(args.out_dir, f"{run}_lime"),
                num_instances=3
            )
        except Exception as e:
            print("LIME falló (continuo):", e)

    with open(os.path.join(args.out_dir, "summary_runs.json"), "w") as f:
        json.dump(runs, f, indent=2)

    best = max(runs, key=lambda r: r["roc_auc"])
    print("\n✅ Mejor modelo:", best)

if __name__ == "__main__":
    main()
