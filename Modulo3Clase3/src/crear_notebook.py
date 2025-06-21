import nbformat as nbf
from pathlib import Path

def crear_jupyter_notebook():
    """
    Crea un archivo notebook/main.ipynb que reproduce el flujo
    de derivación, visualización y optimización de funciones.
    """
    # Ruta base del proyecto
    project_root = Path(__file__).resolve().parent.parent
    notebook_dir = project_root / "notebook"
    notebook_dir.mkdir(exist_ok=True)

    notebook_path = notebook_dir / "main.ipynb"
    nb = nbf.v4.new_notebook()
    nb.cells = []

    # Celda 1: instalación (opcional)
    nb.cells.append(nbf.v4.new_code_cell("""\
# ✅ Instalar dependencias si es necesario
!pip install sympy matplotlib numpy scipy
"""))

    # Celda 2: configurar imports
    nb.cells.append(nbf.v4.new_code_cell("""\
# ✅ Configurar path para importar desde src/
import sys
from pathlib import Path

project_root = Path().resolve().parent
sys.path.append(str(project_root / "src"))
"""))

    # Celda 3: imports
    nb.cells.append(nbf.v4.new_code_cell("""\
# ✅ Importar funciones del proyecto
from sympy import symbols
from derivador import derivar_funcion, encontrar_critico
from visualizador import graficar_funcion_y_derivada
from optimizador import optimizar_funcion
"""))

    # Celda 4: flujo principal
    nb.cells.append(nbf.v4.new_code_cell("""\
# ✅ Ejecutar el análisis completo
x = symbols('x')
f_expr = (x - 3)**2

f_expr, df_expr, x = derivar_funcion(f_expr)
x_critico = encontrar_critico(df_expr, x)

print(f"✓ Punto crítico simbólico: x = {x_critico.evalf()}")

graficar_funcion_y_derivada(f_expr, df_expr, x_critico)

resultado = optimizar_funcion()
print(f"✓ Resultado numérico con SciPy: x = {resultado.x[0]:.4f}, f(x) = {resultado.fun:.4f}")

assert abs(x_critico.evalf() - resultado.x[0]) < 1e-4, "Los resultados no coinciden"
print("✔ Validación exitosa: los resultados coinciden")
"""))

    # Guardar notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"✅ Notebook creado en: {notebook_path}")

if __name__ == "__main__":
    crear_jupyter_notebook()
