import json
import timeit
import matplotlib.pyplot as plt
import random

from utils import busqueda_binaria, busqueda_lineal

# -----------------------------
# 2. 📊 Generación de datos de prueba
# -----------------------------
sizes = [10_000, 100_000, 1_000_000]

# -----------------------------
# 3. ⏱️ Medición del rendimiento
# -----------------------------
tiempo_resultados = {
    "lineal": [],
    "binaria": []
}

tiempo_resultados = {
    "lineal": [],
    "binaria": []
}

print("Tiempos de búsqueda:\n")

for size in sizes:
    datos = list(range(size))

    # Elemento al inicio
    objetivo_inicio = 0
    # Elemento en el medio
    objetivo_medio = size // 2
    # Elemento inexistente
    objetivo_no_existe = -1

    # Medir búsquedas lineales
    t_lin_inicio = timeit.timeit(lambda: busqueda_lineal(datos, objetivo_inicio), number=1)
    t_lin_medio = timeit.timeit(lambda: busqueda_lineal(datos, objetivo_medio), number=1)
    t_lin_inexistente = timeit.timeit(lambda: busqueda_lineal(datos, objetivo_no_existe), number=1)

    # Medir búsquedas binarias
    t_bin_inicio = timeit.timeit(lambda: busqueda_binaria(datos, objetivo_inicio), number=1)
    t_bin_medio = timeit.timeit(lambda: busqueda_binaria(datos, objetivo_medio), number=1)
    t_bin_inexistente = timeit.timeit(lambda: busqueda_binaria(datos, objetivo_no_existe), number=1)

    # Guardar en resultados (promedio opcional o separados)
    tiempo_resultados["lineal"].append({
        "tamaño": size,
        "inicio": t_lin_inicio,
        "medio": t_lin_medio,
        "no_existe": t_lin_inexistente
    })

    tiempo_resultados["binaria"].append({
        "tamaño": size,
        "inicio": t_bin_inicio,
        "medio": t_bin_medio,
        "no_existe": t_bin_inexistente
    })

# Mostrar resultados
for i in range(len(sizes)):
    size = sizes[i]
    print(f"Tamaño: {size}")
    print(f"  Lineal -> inicio: {tiempo_resultados['lineal'][i]['inicio']:.6f}s, medio: {tiempo_resultados['lineal'][i]['medio']:.6f}s, no_existe: {tiempo_resultados['lineal'][i]['no_existe']:.6f}s")
    print(f"  Binaria -> inicio: {tiempo_resultados['binaria'][i]['inicio']:.6f}s, medio: {tiempo_resultados['binaria'][i]['medio']:.6f}s, no_existe: {tiempo_resultados['binaria'][i]['no_existe']:.6f}s\n")

# -----------------------------
# 4. 🌿 Visualización con Matplotlib
# -----------------------------

# Extraer valores por escenario de prueba
lineal_inicio = [res['inicio'] for res in tiempo_resultados['lineal']]
lineal_medio = [res['medio'] for res in tiempo_resultados['lineal']]
lineal_no_existe = [res['no_existe'] for res in tiempo_resultados['lineal']]

binaria_inicio = [res['inicio'] for res in tiempo_resultados['binaria']]
binaria_medio = [res['medio'] for res in tiempo_resultados['binaria']]
binaria_no_existe = [res['no_existe'] for res in tiempo_resultados['binaria']]

# Leer resultados desde JSON
with open("resultados_busquedas.json", "r") as f:
    data = json.load(f)

sizes = [d["tamaño"] for d in data["lineal"]]

# Extraer datos para cada escenario
lineal_inicio = [d["inicio"] for d in data["lineal"]]
lineal_medio = [d["medio"] for d in data["lineal"]]
lineal_no_existe = [d["no_existe"] for d in data["lineal"]]

binaria_inicio = [d["inicio"] for d in data["binaria"]]
binaria_medio = [d["medio"] for d in data["binaria"]]
binaria_no_existe = [d["no_existe"] for d in data["binaria"]]

# Crear gráfico combinado con estilo
plt.figure(figsize=(12, 6))
plt.plot(sizes, lineal_inicio, marker='o', label="Lineal - Inicio", linestyle='solid', color='blue')
plt.plot(sizes, lineal_medio, marker='o', label="Lineal - Medio", linestyle='solid', color='green')
plt.plot(sizes, lineal_no_existe, marker='o', label="Lineal - No Existe", linestyle='solid', color='red')

plt.plot(sizes, binaria_inicio, marker='x', label="Binaria - Inicio", linestyle='dashed', color='blue')
plt.plot(sizes, binaria_medio, marker='x', label="Binaria - Medio", linestyle='dashed', color='green')
plt.plot(sizes, binaria_no_existe, marker='x', label="Binaria - No Existe", linestyle='dashed', color='red')

plt.title("Comparación de Tiempos de Búsqueda (Inicio, Medio, No Existe)")
plt.xlabel("Tamaño de la Lista")
plt.ylabel("Tiempo de Ejecución (segundos)")
plt.legend()
plt.grid(True)
plt.tight_layout()

import os

# Obtener el directorio actual donde se encuentra el script
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Definir el nombre del archivo de salida
nombre_archivo = "grafico_comparacion_6_lineas_estilizado.png"

# Crear la ruta completa de salida
output_path = os.path.join(directorio_actual, nombre_archivo)

# Guardar el gráfico en la ruta deseada
plt.savefig(output_path)
plt.show()