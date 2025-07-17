from pathlib import Path
from nbformat import v4 as nbf

# Contenido del script main.py transformado a notebook
notebook_cells = []

# 1. Encabezado de imports
notebook_cells.append(nbf.new_markdown_cell("# Análisis de Modelos de Clasificación y Regresión\nEste notebook replica la ejecución de `main.py` con visualizaciones y entrenamiento."))
notebook_cells.append(nbf.new_code_cell("""\
import pandas as pd
from src.utils import (
    cargar_dataset,
    preprocesar_datos_clasificacion,
    preprocesar_datos_regresion,
)
from src.entrenamiento import (
    entrenar_random_forest,
    entrenar_xgboost,
    entrenar_elasticnet,
    entrenar_regresion_cuantil,
)
from src.visualizador import (
    graficar_dispersion_cuantiles,
    graficar_mejores_metricas_modelos_regresion,
    visualizar_matriz_confusion,
    visualizar_curva_roc,
    graficar_metricas_comparativas_subplots,
    visualizar_pred_vs_real,
)"""))

# 2. Carga de datos
notebook_cells.append(nbf.new_markdown_cell("## Carga y preprocesamiento"))
notebook_cells.append(nbf.new_code_cell("""\
df = cargar_dataset()
X_train_clf, X_test_clf, y_train_clf, y_test_clf, _, _ = preprocesar_datos_clasificacion(df)
X_train_reg, X_test_reg, y_train_reg, y_test_reg, _ = preprocesar_datos_regresion(df)"""))

# 3. Entrenamiento
notebook_cells.append(nbf.new_markdown_cell("## Entrenamiento de modelos"))
notebook_cells.append(nbf.new_code_cell("""\
resultados_clf = []
resultados_reg = []

rf_clf = entrenar_random_forest(X_train_clf, y_train_clf, X_test_clf, y_test_clf)
resultados_clf.append(rf_clf)
visualizar_matriz_confusion(y_test_clf, rf_clf.get("y_pred"), metodo="RandomForest")
visualizar_curva_roc(y_test_clf, rf_clf.get("y_prob"), metodo="RandomForest")

xgb_clf = entrenar_xgboost(X_train_clf, y_train_clf, X_test_clf, y_test_clf)
resultados_clf.append(xgb_clf)
visualizar_matriz_confusion(y_test_clf, xgb_clf.get("y_pred"), metodo="XGBoost")
visualizar_curva_roc(y_test_clf, xgb_clf.get("y_prob"), metodo="XGBoost")

elastic_reg = entrenar_elasticnet(X_train_reg, y_train_reg, X_test_reg, y_test_reg, n_iter=30)
resultados_reg.append(elastic_reg)
visualizar_pred_vs_real(y_test_reg, elastic_reg.get("y_pred"), metodo="ElasticNet")

cuantil_resultados = entrenar_regresion_cuantil(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
resultados_reg.extend(cuantil_resultados)"""))

# 4. Visualización
notebook_cells.append(nbf.new_markdown_cell("## Visualización de resultados"))
notebook_cells.append(nbf.new_code_cell("""\
graficar_metricas_comparativas_subplots(resultados_clf, resultados_reg)
graficar_mejores_metricas_modelos_regresion(resultados_reg)
graficar_dispersion_cuantiles(resultados_reg)
print("\\n✅ Pipeline completado.")"""))

# Guardar el notebook
notebook_path = Path("outputs/main_pipeline.ipynb")
notebook_path.parent.mkdir(exist_ok=True)
with open(notebook_path, "w") as f:
    f.write(nbf.writes(nbf.new_notebook(cells=notebook_cells)))

notebook_path.name
