import json
import numpy as np
import numpy as np
from transformador_2d import aplicar_transformacion, obtener_matriz_escalado, obtener_matriz_rotacion
from utils_plot import graficar_solucion, graficar_transformacion

from algebra_solver import resolver_sistema

from pathlib import Path


# Ruta absoluta a /outputs en el root del proyecto
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

def guardar_resultado_json(nombre_archivo, datos, output_dir=OUTPUT_DIR):
    """
    Guarda los resultados en formato JSON en la carpeta outputs del proyecto.

    Parámetros:
    - nombre_archivo (str): nombre del archivo sin extensión
    - datos (dict): diccionario de datos a guardar
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # Asegura que la carpeta exista
    path = output_dir / f"{nombre_archivo}.json"

    with open(path, "w") as f:
        json.dump(datos, f, indent=4)

    print(f"✅ Resultado guardado en: {path}")




def ejecutar_transformacion_2d():
    """
    Aplica una transformación lineal 2D (rotación + escalado)
    sobre un conjunto de puntos y grafica el resultado.
    """
    puntos = np.array([
        [1, 2],
        [3, 1],
        [2, 4],
        [4, 3],
        [0, 0],
        [3, 3]
    ], dtype=float)

    angulo = 45
    factor = 1.5

    # Obtener matrices de transformación
    matriz_rotacion = obtener_matriz_rotacion(angulo)
    matriz_escalado = obtener_matriz_escalado(factor)

    # Calcular matriz compuesta: T = S @ R
    T = matriz_escalado @ matriz_rotacion

    # Aplicar transformación
    puntos_transformados = aplicar_transformacion(puntos, T)

    # Construir título y nombre de archivo
    titulo = f"Transformación 2D: Rotación {angulo}° + Escalado {factor}"
    nombre_archivo = f"transformacion_rot{angulo}_esc{factor}.png"

    # Generar gráfico
    graficar_transformacion(
        puntos_originales=puntos,
        puntos_transformados=puntos_transformados,
        titulo=titulo,
        filename=nombre_archivo
    )


def ejecutar_sistemas_lineales():
    """
    Resuelve tres tipos de sistemas lineales:
    cuadrado, sobredeterminado y subdeterminado.
    Guarda resultados como JSON y genera gráficas.

    Retorna:
    - list[dict]: lista de sistemas utilizados
    """
    sistemas = [
        {
            "nombre": "cuadrado",
            "A": np.array([[3, -1, 2], [1, 2, 1], [2, 1, 3]], dtype=float),
            "b": np.array([5, 6, 7], dtype=float)
        },
        {
            "nombre": "sobredeterminado",
            "A": np.array([[1, 2], [2, 1], [3, 4]], dtype=float),
            "b": np.array([5, 6, 7], dtype=float)
        },
        {
            "nombre": "subdeterminado",
            "A": np.array([[1, 2, 3], [4, 5, 6]], dtype=float),
            "b": np.array([7, 8], dtype=float)
        }
    ]

    for sistema in sistemas:
        nombre = sistema["nombre"]
        A, b = sistema["A"], sistema["b"]

        print(f"=== SISTEMA: {nombre.upper()} ===")
        x = resolver_sistema(A, b)

        print("Forma de A:", A.shape)
        print("Solución x:", x, "\n")

        resultado = {
            "tipo": nombre,
            "forma_A": A.shape,
            "solucion": x.tolist()
        }
        guardar_resultado_json(f"solucion_{nombre}", resultado)
        graficar_solucion(x, nombre)

    return sistemas  # ← esto es importante para usarlo luego en main


import numpy as np
import os

from transformador_2d import (
    obtener_matriz_rotacion,
    obtener_matriz_escalado,
    aplicar_transformacion
)
from utils_plot import graficar_transformacion


def ejecutar_transformacion_por_sistema(sistemas, angulo=45, factor=1.5, output_dir="outputs"):
    """
    Aplica una transformación lineal 2D a un conjunto de puntos
    para cada tipo de sistema lineal recibido.

    Parámetros:
    - sistemas (list): Lista de diccionarios con claves 'nombre', 'A', 'b'.
    - angulo (float): Ángulo de rotación en grados.
    - factor (float): Factor de escalado.
    - output_dir (str): Carpeta donde guardar los gráficos.
    """
    puntos = np.array([
        [1, 1],
        [2, 1],
        [2, 2],
        [1, 2]
    ], dtype=float)

    matriz_rotacion = obtener_matriz_rotacion(angulo)
    matriz_escalado = obtener_matriz_escalado(factor)
    T = matriz_escalado @ matriz_rotacion

    for sistema in sistemas:
        nombre = sistema["nombre"]

        puntos_transformados = aplicar_transformacion(puntos, T)

        titulo = f"Transformación 2D: {nombre.capitalize()} ({angulo}° rotación, x{factor} escala)"
        filename = f"transformacion_{nombre.lower()}.png"

        graficar_transformacion(
            puntos_originales=puntos,
            puntos_transformados=puntos_transformados,
            titulo=titulo,
            filename=filename,
            output_dir=output_dir
        )
