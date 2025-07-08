"""
visualizador.py - Funciones para comparar resultados y visualizaciones
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

def mostrar_resultados(resultados):
    """
    Visualiza comparación de F1-scores y tiempo de ejecución con buenas prácticas:
    - Valores numéricos en las barras
    - Inset zoom en F1 para diferencias pequeñas
    - Guardado automático en carpeta outputs
    """

    # Crear carpeta outputs si no existe
    os.makedirs("outputs", exist_ok=True)

    metodos = [r["metodo"] for r in resultados]
    f1_scores = [r["f1"] for r in resultados]
    tiempos = [r["tiempo"] for r in resultados]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # ---------- Gráfico principal F1-score ----------
    bars_f1 = ax[0].bar(metodos, f1_scores, color="skyblue", edgecolor="black")
    ax[0].set_title("F1-Score por Modelo")
    ax[0].set_ylabel("F1-Score")
    ax[0].set_ylim(0.0, 1.0)
    ax[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Agregar valores numéricos sobre las barras
    for bar in bars_f1:
        height = bar.get_height()
        ax[0].annotate(f"{height:.4f}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    # Inset para zoom
    
    axins = inset_axes(ax[0], width="50%", height="45%", loc='upper right')
    axins.bar(metodos, f1_scores, color="skyblue", edgecolor="black")
    axins.set_ylim(min(f1_scores) - 0.005, max(f1_scores) + 0.005)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("Zoom F1", fontsize=8)

    # ---------- Gráfico tiempos ----------
    bars_time = ax[1].bar(metodos, tiempos, color="lightgreen", edgecolor="black")
    ax[1].set_title("Tiempo de Optimización (s)")
    ax[1].set_ylabel("Segundos")
    ax[1].grid(axis="y", linestyle="--", alpha=0.4)

    # Agregar etiquetas de tiempo
    for bar in bars_time:
        height = bar.get_height()
        ax[1].annotate(f"{height:.2f}s",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    # Ajustes finales
    plt.suptitle("Comparación de Modelos: F1-Score y Tiempo", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("outputs/comparacion_modelos_f1_tiempo.png", dpi=300)
    plt.show()


def mostrar_evolucion(skopt_scores, hyperopt_scores, f1_base=None):
    """Grafica la evolución de las F1-scores durante las optimizaciones"""
    plt.figure(figsize=(16, 6))

    plt.plot(skopt_scores, marker="o", label="Scikit-Optimize")
    plt.plot(hyperopt_scores, marker="x", label="Hyperopt")

    if f1_base is not None:
        plt.axhline(y=f1_base, color="gray", linestyle="--", label=f"Base (F1={f1_base:.4f})")

    plt.title("Evolución de F1-score durante la Optimización")
    plt.xlabel("Iteración")
    plt.ylabel("F1-score")
    plt.ylim(min(min(skopt_scores), min(hyperopt_scores), f1_base or 1) - 0.01,
             max(max(skopt_scores), max(hyperopt_scores), f1_base or 1) + 0.01)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/comparacion_resultados.png", bbox_inches="tight")

    plt.show()