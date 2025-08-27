# scripts/main.py
import os, sys, argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

# --- PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports del proyecto ---
from src.data.images import (
    load_flowers_from_dir, load_cifar10, add_preprocessing_and_augmentation, get_backbone_and_preprocess
)
from src.models.tl import build_transfer_model, get_backbone_and_preprocess as _getback
from src.trainer.train_tl import compile_model, train_model, evaluate_and_save
from src.visualizer.plots_tl import plot_history_from_json, plot_confusion_matrix, plot_predictions_grid

def enable_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass
    return gpus

def parse_args():
    p = argparse.ArgumentParser(description="Transfer Learning (ResNet50/EfficientNetB0) en imágenes reales")
    p.add_argument("--dataset", choices=["flowers","cifar10"], default="flowers")
    p.add_argument("--data-dir", default="data/flower_photos")
    p.add_argument("--base", choices=["resnet50","efficientnetb0"], default="resnet50")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--train-base", action="store_true", help="Descongelar backbone")
    p.add_argument("--fine-tune-at", type=int, default=None, help="Capa a partir de la cual entrenar si train_base")
    p.add_argument("--out-dir", default="outputs_tl")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed); tf.random.set_seed(args.seed)

    gpus = enable_memory_growth()
    print("GPUs visibles:", gpus)

    # 1) Datos
    if args.dataset == "flowers":
        ds_train, ds_val, ds_test, num_classes, idx_to_name = load_flowers_from_dir(
            data_dir=args.data_dir, img_size=args.img_size, batch_size=args.batch_size
        )
    else:
        ds_train, ds_val, ds_test, num_classes, idx_to_name = load_cifar10(
            img_size=args.img_size, batch_size=args.batch_size
        )

    # 2) Preproceso específico del backbone (+ augment en train)
    Constructor, preprocess_input, _ = _getback(args.base)
    ds_train = add_preprocessing_and_augmentation(ds_train, preprocess_input, args.img_size, training=True)
    ds_val   = add_preprocessing_and_augmentation(ds_val,   preprocess_input, args.img_size, training=False)
    ds_test  = add_preprocessing_and_augmentation(ds_test,  preprocess_input, args.img_size, training=False)

    # 3) Modelo
    model = build_transfer_model(
        base=args.base,
        num_classes=num_classes,
        input_shape=(args.img_size, args.img_size, 3),
        dropout=args.dropout,
        train_base=args.train_base,
        fine_tune_at=args.fine_tune_at,
    )
    model = compile_model(model, lr=args.lr, loss="sparse_categorical_crossentropy", metrics=("accuracy",))

    run = f"{args.base}__{args.dataset}__img{args.img_size}__bs{args.batch_size}__ep{args.epochs}" \
          + ("__ft" if args.train_base else "")

    # 4) Entrenamiento
    history, ckpt = train_model(
        model, ds_train, ds_val, out_dir=args.out_dir, run_name=run, epochs=args.epochs, patience=2
    )

    # 5) Evaluación
    eval_out = evaluate_and_save(model, ds_test, out_dir=args.out_dir, run_name=run, idx_to_name=idx_to_name)
    print(f"✅ Test accuracy: {eval_out['accuracy']:.4f}")

    # 6) Visualización
    plot_history_from_json(
        os.path.join(args.out_dir, f"{run}_history.json"),
        os.path.join(args.out_dir, f"{run}_curves.png")
    )
    plot_confusion_matrix(
        eval_out["cm"], [idx_to_name[i] for i in range(len(idx_to_name))],
        os.path.join(args.out_dir, f"{run}_cm.png")
    )

    # Muestra grid de predicciones vs reales (tomamos un batch del test)
    for x, y in ds_test.take(1):
        # Ojo: x ya viene preprocesado (imagen normalizada para backbone), para visualizar sin “tintear”
        # tomamos la versión sin preprocess del dataset original o re-normalizamos a [0,1] aproximado.
        # Para simplicidad, mostramos el tensor preprocesado reescalado.
        y_pred = model.predict(x, verbose=0).argmax(axis=1)
        # re-escala simple a [0,1] para mostrar
        x_show = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x) + 1e-8)
        plot_predictions_grid(
            x_show.numpy(), y.numpy(), y_pred,
            idx_to_name, os.path.join(args.out_dir, f"{run}_pred_grid.png"), cols=5, max_items=25
        )
        break

if __name__ == "__main__":
    main()
