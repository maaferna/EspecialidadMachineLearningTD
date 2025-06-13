# Context manager para medir tiempos

import time


class Timer:
    """
    Context manager para medir tiempos de ejecución en secciones críticas del sistema.
    Uso:
        with Timer("Tiempo BFS"):
            resultado = ejecutar_algoritmo()
    """
    def __init__(self, label="⏱️ Tiempo"):
        self.label = label
        self.interval = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print(f"{self.label}: {self.interval:.6f} segundos")
