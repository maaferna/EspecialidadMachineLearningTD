import json
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import Timer, suma_productos_bucle, suma_productos_numba, suma_productos_vectorizada

# Ejecutar mediciones
tamaños = [10_000, 100_000, 1_000_000]
resultados = {"bucle": [], "vectorizado": [], "numba": []}

for n in tamaños:
    a = np.random.rand(n)
    b = np.random.rand(n)

    with Timer(f"Bucle nativo ({n})") as t1:
        suma_productos_bucle(a, b)
    with Timer(f"Vectorizado NumPy ({n})") as t2:
        suma_productos_vectorizada(a, b)
    with Timer(f"Numba JIT ({n})") as t3:
        suma_productos_numba(a, b)

    resultados["bucle"].append(t1.interval)
    resultados["vectorizado"].append(t2.interval)
    resultados["numba"].append(t3.interval)

# Guardar resultados en JSON
output_dir = os.getcwd()
json_path = os.path.join(output_dir, "resultados_optimizacion.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=4, ensure_ascii=False)

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(tamaños, resultados["bucle"], label="Bucle nativo", marker="o")
plt.plot(tamaños, resultados["vectorizado"], label="NumPy vectorizado", marker="s", linestyle="--")
plt.plot(tamaños, resultados["numba"], label="Numba JIT", marker="^", linestyle=":")
plt.title("Comparación de Optimización en Python")
plt.xlabel("Tamaño del arreglo")
plt.ylabel("Tiempo de ejecución (segundos)")
plt.legend()
plt.grid(True)

# Guardar gráfico en el mismo directorio
grafico_path = os.path.join(output_dir, "grafico_optimizacion.png")
plt.savefig(grafico_path)
plt.show()
