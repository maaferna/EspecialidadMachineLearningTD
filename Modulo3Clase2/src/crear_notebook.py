from pathlib import Path
import nbformat as nbf

def crear_jupyter_notebook():
    # 📁 Ruta raíz del proyecto
    project_root = Path(__file__).resolve().parent.parent
    notebook_dir = project_root / "notebook"
    notebook_dir.mkdir(exist_ok=True)

    notebook_path = notebook_dir / "main.ipynb"
    print(f"📝 Generando notebook en: {notebook_path}")
    nb = nbf.v4.new_notebook()
    nb.cells = []

    # ✅ Celda 1: Instalación de dependencias locales
    nb.cells.append(nbf.v4.new_code_cell("""\
# ✅ Instalar dependencias (si es necesario)
!pip install numpy matplotlib
"""))

    # ✅ Celda 2: Configurar sys.path para importar desde src/
    nb.cells.append(nbf.v4.new_code_cell("""\
# ✅ Agregar ruta src/ al path
import sys
from pathlib import Path

project_root = Path().resolve().parent
sys.path.append(str(project_root / "src"))
"""))

    # ✅ Celda 3: Importaciones
    nb.cells.append(nbf.v4.new_code_cell("""\
# ✅ Importar funciones necesarias
from sistemas_lineales import ejecutar_sistemas_lineales
from utils import ejecutar_transformacion_por_sistema
"""))

    # ✅ Celda 4: Ejecutar sistema + transformaciones
    nb.cells.append(nbf.v4.new_code_cell("""\
# ✅ Ejecutar flujo principal
sistemas = ejecutar_sistemas_lineales()
ejecutar_transformacion_por_sistema(sistemas)
"""))

    # ✅ Guardar notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"✅ Notebook generado en: {notebook_path}")

if __name__ == "__main__":
    crear_jupyter_notebook()
