from pathlib import Path
import nbformat as nbf

# ✅ Obtener la ruta real del archivo crear_notebook.py
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

notebooks_dir = project_root / "notebooks"
outputs_dir = project_root / "outputs"

# Crear carpetas necesarias
notebooks_dir.mkdir(parents=True, exist_ok=True)
outputs_dir.mkdir(parents=True, exist_ok=True)

# Crear notebook
nb = nbf.v4.new_notebook()
nb.cells = []

# Celda 1: instalación de dependencias
nb.cells.append(nbf.v4.new_code_cell("""\
# ✅ Instalación de dependencias necesarias
%pip install numpy matplotlib
"""))

# Celda 2: importar funciones desde src/
nb.cells.append(nbf.v4.new_code_cell("""\
# ✅ Configurar path para importar módulos desde src/
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent / 'scripts'))

from utils import generar_datos, ajustar_modelo, graficar_resultado
"""))

# Celda 3: generar datos
nb.cells.append(nbf.v4.new_code_cell("""\
# 🔢 Generar datos sintéticos
x, y = generar_datos()
"""))

# Celda 4: ajustar modelo
nb.cells.append(nbf.v4.new_code_cell("""\
# 📐 Ajustar el modelo con álgebra matricial
beta = ajustar_modelo(x, y)
"""))

# Celda 5: graficar resultados
nb.cells.append(nbf.v4.new_code_cell("""\
# 📊 Graficar resultados y guardar imagen
graficar_resultado(x, y, beta)
"""))

# Guardar notebook
notebook_path = notebooks_dir / "main.ipynb"
with notebook_path.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

# ✅ Mostrar ruta final generada
print(f"✅ Notebook generado en: {notebook_path.resolve()}")
