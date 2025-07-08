import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # Necesario para gráficos 3D en Matplotlib


def graficar_funcion_3d(g, x, y, punto_critico, tipo):
    """
    Grafica la superficie 3D de g(x, y) y el punto crítico con su clasificación.

    Args:
        g (Expr): función simbólica de dos variables
        x (Symbol): variable simbólica x
        y (Symbol): variable simbólica y
        punto_critico (tuple): coordenadas (x, y) del punto crítico
        tipo (str): tipo de punto crítico ("Mínimo local", "Máximo local", "Punto de silla")
    """

    # Convierte la función simbólica g(x, y) en una función numérica para evaluar con NumPy
    g_func = lambdify((x, y), g, modules='numpy')

    # Genera valores en el rango [-5, 5] para x e y
    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)

    # Crea una malla (grid) de coordenadas X, Y para evaluar la superficie Z
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = g_func(X, Y)  # Evalúa g(x, y) en toda la malla

    # Crea figura de Matplotlib con tamaño personalizado
    fig = plt.figure(figsize=(12, 6))

    # Primer subplot: superficie 3D
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)  # Dibuja superficie 3D
    ax.scatter(
        punto_critico[0], punto_critico[1], g_func(*punto_critico),
        color='red', s=50
    )  # Dibuja el punto crítico en rojo
    ax.set_title(f"Superficie g(x, y) con punto crítico ({tipo})")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('g(x, y)')

    # Segundo subplot: mapa de contorno 2D
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contour(X, Y, Z, levels=30, cmap='viridis')  # Dibuja contornos
    ax2.scatter(punto_critico[0], punto_critico[1], color='red')  # Marca el punto crítico
    ax2.set_title('Mapa de contorno')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(contour, ax=ax2)  # Barra de color para indicar niveles

    # Define la ruta para guardar la imagen de salida
    output_path = (
        Path(__file__).resolve().parent.parent / "outputs" / "grafico_3d.png"
    )
    output_path.parent.mkdir(exist_ok=True)  # Crea carpeta outputs si no existe

    # Ajusta el diseño, guarda la figura y cierra el gráfico
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Imprime confirmación de guardado
    print(f"\n📊 Gráfico guardado en: {output_path}")

