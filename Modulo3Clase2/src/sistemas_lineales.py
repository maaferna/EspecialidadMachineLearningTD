# src/sistemas_lineales.py

import numpy as np
from algebra_solver import resolver_sistema
from utils import guardar_resultado_json
from utils_plot import graficar_solucion


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

    return sistemas
