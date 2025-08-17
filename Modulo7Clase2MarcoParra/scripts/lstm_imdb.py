# scripts/lstm_imdb.py
import os
import sys
from pathlib import Path

# ---- PYTHONPATH ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- Imports del proyecto ----
from src.utils.preprocessing_text_imdb import load_imdb_sequences
from src.models.lstm import build_lstm_imdb
from src.evaluator.train_eval import compile_model, train_model
from src.evaluator.eval_metrics import evaluate_on_test_generic
from src.visualizer.plots import plot_history_from_json, save_history_qc_report

# ---- TensorFlow (optimizador + callbacks) ----
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def main(out_dir: str = "outputs_imdb",
         num_words: int = 30000,    # ↑ vocab recomendado
         maxlen: int = 250,         # ↑ contexto
         epochs: int = 10,          # margen para ES
         batch_size: int = 64,      # mejor generalización
         bidirectional: bool = False,
         seed: int = 42):

    os.makedirs(out_dir, exist_ok=True)
    tf.random.set_seed(seed)

    # ------------------------------
    # 1) Datos (pad/trunc a maxlen)
    # ------------------------------
    (x_train, y_train), (x_test, y_test), _ = load_imdb_sequences(
        num_words=num_words, maxlen=maxlen
    )

    # ------------------------------
    # 2) Modelo LSTM (regularizado)
    # ------------------------------
    model = build_lstm_imdb(
        vocab_size=num_words,
        maxlen=maxlen,
        embed_dim=128,
        lstm_units=32,        # ↓ capacidad para reducir gap
        bidirectional=bidirectional,
        dropout=0.5,          # + regularización
        rdropout=0.3,
        l2_coef=1e-4          # L2 un poco más fuerte
    )

    # Optimizador más estable (mitiga serrucho en val_loss)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, clipnorm=1.0)

    # Compilar
    model = compile_model(
        model, loss="binary_crossentropy",
        optimizer=optimizer, metrics=["accuracy"]
    )

    tag_bi = "__bi" if bidirectional else ""
    run_name = f"imdb_lstm__adam2e-4_clip1.0__bce__vs0.2__bs{batch_size}__ep{epochs}{tag_bi}"

    # -------------------------------------------
    # 3) Callbacks (guardar mejor + ES + ReduceLR)
    # -------------------------------------------
    ckpt_path = os.path.join(out_dir, f"{run_name}_best.weights.h5")
    callbacks = [
        # Guarda SIEMPRE los mejores PESOS por val_loss
        ModelCheckpoint(ckpt_path, monitor="val_loss",
                        save_best_only=True, save_weights_only=True, verbose=1),
    ]
    if epochs >= 5:
        callbacks += [
            EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1,
                              min_lr=1e-5, verbose=1),
        ]

    # ------------------------------
    # 4) Entrenar (history a JSON)
    # ------------------------------
    history = train_model(
        model, x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        out_dir=out_dir,
        run_name=run_name,
        callbacks=callbacks    # se reenvía a model.fit(**fit_kwargs)
    )

    # 🔁 Cargar explícitamente los MEJORES pesos por val_loss
    if os.path.isfile(ckpt_path):
        model.load_weights(ckpt_path)

    # ------------------------------
    # 5) Evaluación en test (binario)
    # ------------------------------
    test_acc, _ = evaluate_on_test_generic(
        model, x_test, y_test, num_classes=2,
        out_dir=out_dir, run_name=run_name
    )
    print(f"✅ IMDb Test accuracy (best weights): {test_acc:.4f}")

    # ------------------------------
    # 6) Curvas + QC
    # ------------------------------
    plot_history_from_json(
        os.path.join(out_dir, f"{run_name}_history.json"),
        os.path.join(out_dir, f"{run_name}_curves.png"),
    )

    # Rango razonable para este preset
    save_history_qc_report(
        history, test_acc,
        out_path=os.path.join(out_dir, f"{run_name}_history_qc.md"),
        expected_range=(0.86, 0.89)
    )

if __name__ == "__main__":
    main()
