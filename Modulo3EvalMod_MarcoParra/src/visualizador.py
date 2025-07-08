from matplotlib import pyplot as plt
# ---------------------------
# 5. Visualización de Resultados
# ---------------------------
from pathlib import Path

def visualizar_resultados(costs_gd, costs_sgd, ws_gd, bs_gd, ws_sgd, bs_sgd,
                          output_path=Path("notebook/outputs/comparacion_gd_sgd.png")):
    """
    Visualiza y guarda una comparación entre los resultados de GD y SGD.
    """
    # Crear carpeta si no existe
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Crear figura
    plt.figure(figsize=(12, 5))

    # Costo
    plt.subplot(1, 2, 1)
    plt.plot(costs_gd, label="GD")
    plt.plot(costs_sgd, label="SGD")
    plt.title("Costo MSE")
    plt.xlabel("Iteración")
    plt.ylabel("Error cuadrático medio")
    plt.legend()

    # Parámetros
    plt.subplot(1, 2, 2)
    plt.plot(ws_gd, label="w (GD)")
    plt.plot(bs_gd, label="b (GD)")
    plt.plot(ws_sgd, label="w (SGD)")
    plt.plot(bs_sgd, label="b (SGD)")
    plt.title("Evolución de parámetros")
    plt.xlabel("Iteración / Época")
    plt.ylabel("Valor")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n📊 Gráfico guardado en: {output_path}")
    plt.close()

