from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

# Crear notebook
nb = new_notebook()
cells = []

# Celda 1: Instalar dependencias
cells.append(new_code_cell("""\
# ✅ Instalar dependencias necesarias (si no usas conda)
!pip install -q pandas numpy scikit-learn matplotlib seaborn optuna
"""))

# Celda 2: Ajustar path para importar desde src/
cells.append(new_code_cell("""\
# ✅ Ajustar path para importar desde src/
import sys
from pathlib import Path

# ✅ Ir un solo nivel hacia arriba (de notebook/ a la raíz del proyecto)
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
"""))

# Celda 3: Importaciones del proyecto
cells.append(new_code_cell("""\
# ✅ Importar funciones del pipeline
from src.utils import cargar_dataset, preprocesar_datos, entrenar_modelo_base
from src.optimizacion import (
    optimizar_con_gridsearch,
    optimizar_con_randomsearch,
    optimizar_con_optuna,
)
from src.visualizador import (
    visualizar_matriz_confusion,
    visualizar_curva_roc,
    graficar_metricas_comparativas,
)
"""))

# Celda 4: Cargar y procesar datos
cells.append(new_code_cell("""\
# 🚀 Iniciar pipeline
df = cargar_dataset()
X_train, X_test, y_train, y_test = preprocesar_datos(df)
"""))

# Celda 5: Modelo base
cells.append(new_code_cell("""\
# 🌲 Modelo base
resultados = []
resultado_base = entrenar_modelo_base(X_train, y_train, X_test, y_test)
resultados.append({
    "metodo": "Base",
    "f1": resultado_base["f1"],
    "precision": resultado_base["precision"],
    "recall": resultado_base["recall"],
    "auc": resultado_base["auc"]
})
visualizar_matriz_confusion(y_test, resultado_base["y_pred"], metodo="Base")
visualizar_curva_roc(y_test, resultado_base["y_prob"], metodo="Base")
"""))

# Celda 6: GridSearch
cells.append(new_code_cell("""\
# 🔍 GridSearch
resultado_grid = optimizar_con_gridsearch(X_train, y_train, X_test, y_test)
resultados.append({
    "metodo": "GridSearch",
    "f1": resultado_grid["f1"],
    "precision": resultado_grid["precision"],
    "recall": resultado_grid["recall"],
    "auc": resultado_grid["auc"]
})
visualizar_matriz_confusion(y_test, resultado_grid["y_pred"], metodo="GridSearch")
visualizar_curva_roc(y_test, resultado_grid["y_prob"], metodo="GridSearch")
"""))

# Celda 7: RandomSearch
cells.append(new_code_cell("""\
# 🎲 RandomSearch
resultado_random = optimizar_con_randomsearch(X_train, y_train, X_test, y_test)
resultados.append({
    "metodo": "RandomSearch",
    "f1": resultado_random["f1"],
    "precision": resultado_random["precision"],
    "recall": resultado_random["recall"],
    "auc": resultado_random["auc"]
})
visualizar_matriz_confusion(y_test, resultado_random["y_pred"], metodo="RandomSearch")
visualizar_curva_roc(y_test, resultado_random["y_prob"], metodo="RandomSearch")
"""))

# Celda 8: Optuna
cells.append(new_code_cell("""\
# 🔮 Optuna
resultado_optuna = optimizar_con_optuna(X_train, y_train, X_test, y_test)
resultados.append({
    "metodo": "Optuna",
    "f1": resultado_optuna["f1"],
    "precision": resultado_optuna["precision"],
    "recall": resultado_optuna["recall"],
    "auc": resultado_optuna["auc"]
})
visualizar_matriz_confusion(y_test, resultado_optuna["y_pred"], metodo="Optuna")
visualizar_curva_roc(y_test, resultado_optuna["y_prob"], metodo="Optuna")
"""))

# Celda 9: Comparación final
cells.append(new_code_cell("""\
# 📊 Comparación de todos los modelos
graficar_metricas_comparativas(resultados)
"""))

# Guardar notebook
nb.cells = cells
output_path = Path("notebooks/diabetes_pipeline.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

output_path
