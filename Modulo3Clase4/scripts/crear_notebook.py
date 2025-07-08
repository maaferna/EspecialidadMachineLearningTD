"""
Este script genera automáticamente un notebook de Jupyter
para el análisis simbólico y visual de una función de dos variables.

Incluye instalación automática de dependencias en la primera celda (via pip).
"""
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from pathlib import Path


def crear_notebook(nombre="main.ipynb"):
    """
    Crea un Jupyter Notebook con el flujo completo del análisis simbólico y visual de una función de dos variables.
    Incluye la instalación opcional de dependencias y ajustes del path para importar desde src/.
    """
    notebook = new_notebook()
    cells = []

    # Celda 0: Instalar dependencias con pip
    cells.append(new_code_cell("""\
# ✅ Instalación de dependencias (solo si no usas conda)
!pip install sympy numpy matplotlib nbformat"""))

    # Celda 1: Ajustar el path para importar desde src/
    cells.append(new_code_cell("""\
# ✅ Registrar src/ como ruta de importación
import sys
from pathlib import Path
import os

project_root = Path().resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))
                               """))

    # Celda 2: Importaciones
    cells.append(new_code_cell("""\
# ✅ Importar módulos personalizados y símbolos necesarios
from funciones import definir_funcion
from utils import calcular_gradiente, calcular_hessiana, encontrar_punto_critico
from visualizador import graficar_funcion_3d
from clasificador import clasificar_punto_critico
                               """))

    # Celda 3: Definir función
    cells.append(new_code_cell("""\
# ✅ Definir la función simbólica g(x, y)
g, x, y = definir_funcion()"""))

    # Celda 4: Calcular gradiente y Hessiana
    cells.append(new_code_cell("""\
# ✅ Calcular gradiente ∇g y matriz Hessiana H
grad = calcular_gradiente(g, x, y)
hess = calcular_hessiana(g, x, y)
print("Gradiente:", grad)
print("Hessiana:", hess)"""))

    # Celda 5: Punto crítico
    cells.append(new_code_cell("""\
# ✅ Resolver ∇g(x, y) = (0, 0)
puntos_criticos = encontrar_punto_critico(grad, x, y)
punto = puntos_criticos[0]
punto"""))

    # Celda 6: Clasificación del punto crítico
    cells.append(new_code_cell("""\
# ✅ Clasificar el punto crítico
tipo = clasificar_punto_critico(hess, punto)
tipo"""))

    # Celda 7: Visualización
    cells.append(new_code_cell("""\
# ✅ Visualizar g(x, y) y el punto crítico
graficar_funcion_3d(g, x, y, (punto[x], punto[y]), tipo)"""))

    # Celda 8: Mostrar imagen desde archivo guardado
    cells.append(new_code_cell("""\
# ✅ Mostrar gráfico guardado desde outputs/
from IPython.display import Image, display
img_path = Path().resolve().parent / "outputs" / "grafico_3d.png"
display(Image(filename=str(img_path)))"""))

    # Guardar notebook
    output_path = Path(__file__).resolve().parent.parent / "notebooks" / nombre
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook.cells = cells

    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    print(f"✅ Notebook generado: {output_path}")


if __name__ == "__main__":
    crear_notebook()
