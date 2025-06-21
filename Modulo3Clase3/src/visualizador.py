import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify
from pathlib import Path


def graficar_funcion_y_derivada(f_expr, df_expr, x_critico):
    """
    Grafica f(x) y su derivada f'(x), marcando el punto crítico.

    Args:
        f_expr (Expr): expresión simbólica de f(x)
        df_expr (Expr): derivada de f(x)
        x_critico (float): punto donde se anula la derivada
    """
    # Generar un conjunto de valores de x para evaluar
    x_vals = np.linspace(-5, 10, 400)
    print(f"📊 Generando valores de x desde -5 hasta 10 con 400 puntos...")
    print(x_vals)
    
    print(f"\n📈 Generando gráfico de f(x) y su derivada f'(x)...")
    # Convertir expresiones simbólicas en funciones numéricas
    f_num = lambdify('x', f_expr, 'numpy')
    df_num = lambdify('x', df_expr, 'numpy')

    y_vals = f_num(x_vals)
    dy_vals = df_num(x_vals)

    # Crear figura
    plt.figure(figsize=(10, 5))

    # Graficar f(x)
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')

    # Graficar f'(x)
    plt.plot(x_vals, dy_vals, label="f'(x)", color='green')

    # Línea vertical y punto en el mínimo
    plt.axvline(float(x_critico), linestyle='--', color='red',
                label=f'Punto crítico x = {float(x_critico)}')
    plt.scatter([float(x_critico)], [f_num(float(x_critico))], color='red')

    # Personalización del gráfico
    plt.title('f(x) y su derivada')
    plt.xlabel('x')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)

    # Ruta de salida
    output_path = Path(__file__).resolve().parent.parent / "outputs" / "funcion_derivada.png"
    output_path.parent.mkdir(exist_ok=True)

    # Guardar y cerrar figura
    plt.savefig(output_path)
    plt.close()
    print(f"\n📊 Gráfico guardado en: {output_path}")
