import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from pathlib import Path

# Inicializa el notebook
nb = new_notebook()
cells = []

# Celda 1: Instalación de dependencias
cells.append(new_code_cell("""\
# ✅ Instalar dependencias si estás fuera de un entorno virtual Conda
!pip install -q pandas numpy matplotlib seaborn scikit-learn optuna scikit-optimize hyperopt nbformat
"""))

# Celda 2: Ajustar sys.path para importar desde src/
cells.append(new_code_cell("""\
import sys
from pathlib import Path

project_root = Path().resolve().parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))
"""))

# Celda 3: Importaciones
cells.append(new_code_cell("""\
import matplotlib.pyplot as plt
from utils import cargar_dataset, preprocesar_datos
from modelos import entrenar_modelo_base
from optimizacion import (
    optimizar_con_gridsearch,
    optimizar_con_randomsearch,
)
from visualizador import (
    visualizar_matriz_confusion,
    visualizar_curva_roc,
    mostrar_resultados_comparativos,
)
"""))

# Celda 4: Ejecución del pipeline
cells.append(new_code_cell("""\
print("\\n🚀 Iniciando pipeline de clasificación de diabetes...")

df = cargar_dataset()
X_train, X_test, y_train, y_test = preprocesar_datos(df)

resultados = []

# Modelo base
resultado_base = entrenar_modelo_base(X_train, y_train, X_test, y_test)
resultados.append(resultado_base)
visualizar_matriz_confusion(y_test, resultado_base["y_pred"], metodo="Base")
visualizar_curva_roc(y_test, resultado_base["y_prob"], metodo="Base")

# Grid Search
resultado_grid = optimizar_con_gridsearch(X_train, y_train, X_test, y_test)
resultados.append(resultado_grid)
visualizar_matriz_confusion(y_test, resultado_grid["y_pred"], metodo="GridSearch")
visualizar_curva_roc(y_test, resultado_grid["y_prob"], metodo="GridSearch")

# Random Search
resultado_random = optimizar_con_randomsearch(X_train, y_train, X_test, y_test)
resultados.append(resultado_random)
visualizar_matriz_confusion(y_test, resultado_random["y_pred"], metodo="RandomSearch")
visualizar_curva_roc(y_test, resultado_random["y_prob"], metodo="RandomSearch")

# Comparación final
mostrar_resultados_comparativos(resultados)

# Reflexión
print("\\n📌 Reflexión Final:")
print("- GridSearch suele encontrar buenos modelos pero puede ser más lento.")
print("- RandomSearch explora más y puede ser más eficiente con menos combinaciones.")
print("- Comparar tiempos, F1 y AUC para definir cuál se adapta mejor al problema.")

print("\\n✅ Pipeline completado.")
"""))

# Guardar notebook
notebook_path = Path(__file__).resolve().parent.parent / "notebooks" / "diabetes_clasificacion.ipynb"
notebook_path.parent.mkdir(parents=True, exist_ok=True)
nb.cells = cells

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"✅ Notebook generado en: {notebook_path}")
