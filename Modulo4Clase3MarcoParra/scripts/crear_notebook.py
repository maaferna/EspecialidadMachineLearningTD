import nbformat as nbf
from pathlib import Path

# Crear notebook
nb = nbf.v4.new_notebook()
cells = []

# 🔹 0. Título e introducción
cells.append(nbf.v4.new_markdown_cell("# Optimización Bayesiana en Modelos de Clasificación\nEste notebook aplica Scikit-Optimize y Hyperopt para optimizar un modelo Random Forest sobre datos de cáncer de mama."))

# 🔹 1. Instalación de dependencias (opcional para entornos sin environment.yml)
cells.append(nbf.v4.new_code_cell("""\
# ✅ Instala las dependencias necesarias si no estás usando environment.yml
!pip install scikit-learn scikit-optimize hyperopt matplotlib seaborn pandas nbformat
"""))

# 🔹 2. Configuración de paths e importaciones
cells.append(nbf.v4.new_code_cell("""\
import sys
from pathlib import Path

# ✅ Ir un solo nivel hacia arriba (de notebook/ a la raíz del proyecto)
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import (
    cargar_datos,
    entrenar_modelo_base,
    optimizar_con_skopt,
    optimizar_con_hyperopt,
)
from src.visualizador import mostrar_resultados, mostrar_evolucion
"""))

# 🔹 3. Ejecución de entrenamiento y optimización
cells.append(nbf.v4.new_code_cell("""\
print("🔍 Cargando y preparando datos...")
X_train, X_test, y_train, y_test = cargar_datos()
resultados = []

print("🌲 Entrenando modelo base (sin optimización)...")
resultado_base = entrenar_modelo_base(X_train, y_train, X_test, y_test)
resultados.append(resultado_base)

print("\\n⚙️ Optimización con Scikit-Optimize...")
resultado_skopt = optimizar_con_skopt(X_train, y_train, X_test, y_test)
resultados.append(resultado_skopt)

print("\\n🔧 Optimización con Hyperopt...")
resultado_hyperopt = optimizar_con_hyperopt(X_train, y_train, X_test, y_test)
resultados.append(resultado_hyperopt)
"""))

# 🔹 4. Visualización de resultados
cells.append(nbf.v4.new_code_cell("""\
print("\\n📊 Comparación de resultados:")
mostrar_resultados(resultados)

mostrar_evolucion(
    skopt_scores=resultado_skopt["evolucion"],
    hyperopt_scores=resultado_hyperopt["evolucion"],
    f1_base=resultado_base["f1"]
)
"""))

# Guardar notebook
nb['cells'] = cells
notebook_path = Path("notebook/optimizacion_interactiva.ipynb")
notebook_path.parent.mkdir(exist_ok=True)
with open(notebook_path, "w") as f:
    nbf.write(nb, f)

notebook_path.name
