import sys
from pathlib import Path

# Agregar src/ al path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from src.optimizadores import gradient_descent, stochastic_gradient_descent
from src.visualizador import visualizar_resultados
from src.utils import generar_datos_sinteticos
from src.modelo_analitico import calculo_cerrado_regresion


# ---------------------------
# 6. Main
# ---------------------------
if __name__ == "__main__":
    print("Comparación de Gradient Descent (GD) y Stochastic Gradient Descent (SGD)")
    print("Generando datos sintéticos...")
    X, y = generar_datos_sinteticos(n=100, seed=42)

    print("\nResolviendo con cálculo cerrado...")
    try:
        w_cerrado, b_cerrado = calculo_cerrado_regresion(X, y)
        print(f"✅ Parámetros analíticos: w = {w_cerrado:.4f}, b = {b_cerrado:.4f}")
    except Exception as e:
        print(f"⚠️ No se pudo calcular la solución cerrada: {e}")


    print("Datos generados. Iniciando entrenamiento...")

    print("Entrenando con Gradient Descent...")
    print("--------------------------------------------------")
    w_gd, b_gd, costs_gd, ws_gd, bs_gd = gradient_descent(X, y, lr=0.01)
    print("Entrenando con Stochastic Gradient Descent...")
    print("--------------------------------------------------")
    w_sgd, b_sgd, costs_sgd, ws_sgd, bs_sgd = stochastic_gradient_descent(X, y, lr=0.001)

    print("Entrenamiento completado. Visualizando resultados...")
    visualizar_resultados(costs_gd, costs_sgd, ws_gd, bs_gd, ws_sgd, bs_sgd)

    print("\nParámetros finales con GD:", f"w = {w_gd}, b = {b_gd}")
    print("Parámetros finales con SGD:", f"w = {w_sgd}, b = {b_sgd}")
