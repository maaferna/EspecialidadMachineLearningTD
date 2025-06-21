import sys
from pathlib import Path
from sympy import symbols, diff, Eq, solve

# Agregar carpeta src al path para importar módulos
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from derivador import derivar_funcion, encontrar_critico
from visualizador import graficar_funcion_y_derivada
from optimizador import optimizar_funcion

def main():
    """Ejecuta el flujo principal del proyecto."""
    # Definir la función a analizar
    x = symbols('x')
    f_expr = (x - 3)**2

    # Derivar la función pasada
    f_expr, df_expr, x = derivar_funcion(f_expr)
    x_critico = encontrar_critico(df_expr, x)

    print(f"\n✓ Punto crítico simbólico: x = {x_critico.evalf()}")

    graficar_funcion_y_derivada(f_expr, df_expr, x_critico)

    resultado = optimizar_funcion()
    print(f"\n✓ Resultado numérico con SciPy: x = {resultado.x[0]:.4f}, f(x) = {resultado.fun:.4f}")

    assert abs(x_critico.evalf() - resultado.x[0]) < 1e-4, "Los resultados no coinciden"
    print("\n✔ Validación exitosa: los resultados coinciden")



if __name__ == "__main__":
    main()