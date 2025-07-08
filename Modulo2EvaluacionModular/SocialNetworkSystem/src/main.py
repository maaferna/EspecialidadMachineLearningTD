# Reejecutar el bloque completo tras reinicio del estado

import json
import matplotlib.pyplot as plt
from time import time
import numpy as np
from pathlib import Path

from utils.exceptions import UsuarioExistenteError, UsuarioNoEncontradoError
from utils.data_generator import generar_red_social
from models.network import RedSocial
from optimizations.optimized_ops import (
    calcular_amigos_en_comun,
    calcular_amigos_en_comun_numba,
    convertir_amigos_a_numpy
)
from algorithms.bfs import sugerencias_amistad

# Directorio para guardar resultados
output_dir = Path(__file__).resolve().parent.parent / "outputs"

output_dir.mkdir(exist_ok=True)

# Inicializar contenedor para resultados
resultados = []

# Parámetros de prueba
escenarios = [10_000, 50_000, 100_000]

for n_usuarios in escenarios:
    print(f"\n🧪 Ejecutando prueba con {n_usuarios} usuarios")

    red_social = RedSocial()
    red_dict = generar_red_social(n_usuarios=n_usuarios, max_amigos=100)

    for usuario in red_dict:
        try:
            red_social.agregar_usuario(usuario)
        except UsuarioExistenteError:
            continue

    for usuario, amigos in red_dict.items():
        for amigo in amigos:
            try:
                red_social.conectar_usuarios(usuario, amigo)
            except UsuarioNoEncontradoError:
                continue

    red = red_social.obtener_red()
    usuarios = list(red.keys())
    user_a, user_b = usuarios[0], usuarios[1]

    amigos_a = red[user_a]
    amigos_b = red[user_b]
    amigos_np = convertir_amigos_a_numpy({user_a: amigos_a, user_b: amigos_b})

    tiempos = {"usuarios": n_usuarios, "conjuntos": [], "numba": []}

    for i in range(3):
        print(f"\n🔁 Iteración {i+1} con {n_usuarios} usuarios:")

        t0 = time()
        comunes_set = calcular_amigos_en_comun(amigos_a, amigos_b)
        t1 = time()
        tiempos["conjuntos"].append(t1 - t0)
        print(f"Amigos en común (conjuntos): {t1 - t0:.6f} segundos")

        t2 = time()
        comunes_numba = calcular_amigos_en_comun_numba(amigos_np[user_a], amigos_np[user_b])
        t3 = time()
        tiempos["numba"].append(t3 - t2)
        print(f"Amigos en común (Numba): {t3 - t2:.6f} segundos")

    resultados.append(tiempos)

    if n_usuarios == 10_000:
        print(f"\n🔍 Sugerencias de amistad para '{user_a}':")
        try:
            sugerencias = sugerencias_amistad(red_social, user_a, max_sugerencias=5)
            for idx, sugerido in enumerate(sugerencias, start=1):
                print(f"{idx}. {sugerido}")
        except ValueError as e:
            print(f"Error: {e}")

# Guardar resultados como JSON
json_path = output_dir / "resultados_tiempos.json"
with open(json_path, "w") as f:
    json.dump(resultados, f, indent=4)

# Graficar resultados
usuarios = [r["usuarios"] for r in resultados]
conjuntos_prom = [np.mean(r["conjuntos"]) for r in resultados]
numba_prom = [np.mean(r["numba"]) for r in resultados]

plt.figure(figsize=(10, 6))
plt.plot(usuarios, conjuntos_prom, marker='o', label='Set (conjuntos)', linestyle='--')
plt.plot(usuarios, numba_prom, marker='s', label='Numba', linestyle='-')
plt.title("Comparación de Tiempos - Conjuntos vs Numba")
plt.xlabel("Número de Usuarios")
plt.ylabel("Tiempo Promedio (segundos)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "grafico_comparacion_tiempos.png")
plt.show()


