import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from pathlib import Path

# Crear notebook
nb = new_notebook()
cells = []

# Celda 1: Instalar dependencias
cells.append(new_code_cell("""\
# ✅ Instalar dependencias si estás fuera de un entorno virtual
!pip install numpy matplotlib
"""))

# Celda 2: Ajustar sys.path para importar desde src/
cells.append(new_code_cell("""\
# ✅ Ir un solo nivel hacia arriba (de notebook/ a la raíz del proyecto)
import sys
from pathlib import Path

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
"""))

# Celda 3: Importaciones
cells.append(new_code_cell("""\
# ✅ Importar funciones del proyecto
from src.optimizadores import gradient_descent, stochastic_gradient_descent
from src.visualizador import visualizar_resultados
from src.utils import generar_datos_sinteticos
from src.modelo_analitico import calculo_cerrado_regresion
"""))

# Celda 4: Ejecución principal
cells.append(new_code_cell("""\
# ✅ Comparación de GD vs SGD
print("Generando datos sintéticos...")
X, y = generar_datos_sinteticos(n=100, seed=42)

print("\\nCálculo cerrado:")
try:
    w_cerrado, b_cerrado = calculo_cerrado_regresion(X, y)
    print(f"✅ Parámetros analíticos: w = {w_cerrado:.4f}, b = {b_cerrado:.4f}")
except Exception as e:
    print(f"⚠️ Error en cálculo cerrado: {e}")

print("\\nEntrenando con Gradient Descent...")
w_gd, b_gd, costs_gd, ws_gd, bs_gd = gradient_descent(X, y, lr=0.01)

print("Entrenando con Stochastic Gradient Descent...")
w_sgd, b_sgd, costs_sgd, ws_sgd, bs_sgd = stochastic_gradient_descent(X, y, lr=0.001)

print("\\nVisualizando resultados...")
visualizar_resultados(costs_gd, costs_sgd, ws_gd, bs_gd, ws_sgd, bs_sgd)

print(f"GD final: w = {w_gd}, b = {b_gd}")
print(f"SGD final: w = {w_sgd}, b = {b_sgd}")
"""))

# Guardar notebook en notebook/
output_path = Path(__file__).resolve().parent.parent / "notebook" / "regresion_lineal.ipynb"
output_path.parent.mkdir(parents=True, exist_ok=True)
nb.cells = cells

with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"✅ Notebook generado en: {output_path}")
