import os
import matplotlib.pyplot as plt

from transformador_2d import aplicar_transformacion, obtener_matriz_escalado, obtener_matriz_rotacion

def graficar_solucion(x, tipo, output_dir="outputs"):
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(x)), x)
    plt.title(f"Solución del sistema: {tipo}")
    plt.xlabel("Índice de variable")
    plt.ylabel("Valor")
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"grafico_solucion_{tipo}.png")
    plt.savefig(path)
    plt.close()
    print(f"📈 Gráfico guardado en: {path}")


def graficar_transformacion(puntos_originales, puntos_transformados, titulo="Transformación 2D", output_dir="outputs", filename="transformacion_2d.png"):
    """
    Grafica los puntos originales y transformados en 2D.

    Parámetros:
    - puntos_originales (ndarray): Puntos originales (n, 2).
    - puntos_transformados (ndarray): Puntos transformados (n, 2).
    - titulo (str): Título del gráfico.
    - output_dir (str): Carpeta donde guardar la imagen.
    - filename (str): Nombre del archivo de salida.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for (x0, y0), (x1, y1) in zip(puntos_originales, puntos_transformados):
        plt.plot([x0, x1], [y0, y1], 'gray', linestyle='dotted')
        plt.plot(x0, y0, 'bo')
        plt.plot(x1, y1, 'ro')

    plt.title(titulo)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Gráfico de transformación guardado en: {output_path}")
