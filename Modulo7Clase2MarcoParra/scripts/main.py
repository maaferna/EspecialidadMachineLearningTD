# scripts/main.py
# ============================================================
# Runner único (GRID SIEMPRE) para cumplir el enunciado:
#   - Activaciones: (relu, relu) y (relu, tanh)
#   - Pérdidas: categorical_crossentropy, mse
#   - Optimizadores: adam, sgd(momentum=0.9, lr=0.01)
# NOTA: Este archivo NO grafica. Toda la visualización/QC se delega
#       a src/visualizer/plots a través de src/experiments/grid.run_grid
# Artefactos en outputs/:
#   - <run>_history.json / <run>_curves.png
#   - <run>_test_report.json / <run>_confusion_matrix.csv
#   - experiments_summary.csv / experiments_comparison.png
#   - reflection_enunciado.md
# ============================================================

import os
import sys
from pathlib import Path

# --- PYTHONPATH -> agrega raíz del proyecto (para importar 'src.*') ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports del proyecto ---
from src.utils.preprocessing import load_preprocess
from src.experiments.grid import run_grid  # el grid recibe los datos y orquesta todo

def main(out_dir: str = "outputs",
         epochs: int = 10,
         batch_size: int = 32,
         val_split: float = 0.2):
    """Ejecuta el grid completo del enunciado.
    - Carga y preprocesa Fashion-MNIST una sola vez.
    - Llama a run_grid(...) que entrena, evalúa y genera artefactos/plots/QC.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Datos (carga + normalización + one-hot) — una sola vez
    (x_train, y_train), (x_test, y_test) = load_preprocess(
        dataset="fashion",
        one_hot=True,
        normalize=True
    )

    # 2) Ejecutar GRID (visualización/QC se hace dentro de run_grid)
    results = run_grid(
        data=(x_train, y_train, x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        out_dir=out_dir,
        expected_range=(0.88, 0.92)  # típico para MLP en Fashion-MNIST
    )

    # 3) (Opcional) imprimir un breve resumen en consola
    if results:
        best = max(results, key=lambda r: r["test_acc"])
        print(f"✅ Mejor combinación: {best['loss']} + {best['opt']} | "
              f"activaciones=({best['act1']},{best['act2']}) | "
              f"test_acc={best['test_acc']:.4f}")

if __name__ == "__main__":
    main()
