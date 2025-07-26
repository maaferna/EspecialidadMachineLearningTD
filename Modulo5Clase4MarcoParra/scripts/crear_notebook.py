from nbformat import v4 as nbf
from pathlib import Path

# Crear celdas para el notebook
cells = []

# Celda 1: instalación de dependencias
cells.append(nbf.new_code_cell(
    """\
# Requisitos para ejecutar este notebook
!pip install matplotlib seaborn scikit-learn pandas
"""
))

# Celda 2: configuración inline de gráficos
cells.append(nbf.new_code_cell(
    """\
# Configuración para mostrar gráficos en el notebook
%matplotlib inline
"""
))

# Celda 3: importaciones
cells.append(nbf.new_code_cell(
    """\
from sklearn.exceptions import UndefinedMetricWarning
from src.utils import cargar_dataset, preprocesar_datos
import warnings
from src.modelos import (
    crear_modelo_random_forest,
    crear_modelo_logistic_regression,
    crear_modelo_svc,
)
from src.validacion import (
    obtener_kfold,
    obtener_loocv,
    obtener_stratified_kfold,
)
from src.evaluador import evaluar_metricas
from src.visualizador import (
    graficar_metricas_individuales,
    plot_metricas_comparativas
)

# Suprimir advertencias de métricas indefinidas
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
"""
))

# Celda 4: carga y preprocesamiento
cells.append(nbf.new_code_cell(
    """\
# ================================
# Carga y preprocesamiento
# ================================
df = cargar_dataset()
X, y, _ = preprocesar_datos(df)
"""
))

# Celda 5: definición de modelos
cells.append(nbf.new_code_cell(
    """\
# ================================
# Modelos
# ================================
modelos = {
    "RandomForest": crear_modelo_random_forest(),
    "LogisticRegression": crear_modelo_logistic_regression(),
    # "SVC": crear_modelo_svc()
}
"""
))

# Celda 6: definición de validadores
cells.append(nbf.new_code_cell(
    """\
# ================================
# Técnicas de Validación Cruzada
# ================================
validaciones = {
    "KFold": obtener_kfold(n_splits=5),
    # "LeaveOneOut": obtener_loocv(X.values, y.values, n_samples=1000)[0],
    "StratifiedKFold": obtener_stratified_kfold(n_splits=5, random_state=42)
}
"""
))

# Celda 7: evaluación + gráficas individuales
cells.append(nbf.new_code_cell(
    """\
# ================================
# Evaluación + Gráficas Individuales
# ================================
resultados = []

for nombre_modelo, modelo in modelos.items():
    for nombre_validacion, validador in validaciones.items():
        print(f"\\n🔍 Evaluando {nombre_modelo} con {nombre_validacion}...")

        # Evaluación métrica
        metricas = evaluar_metricas(modelo, X.values, y.values, validador)
        metricas["modelo"] = nombre_modelo
        metricas["estrategia"] = nombre_validacion
        resultados.append(metricas)

        # Dividir manualmente para visualizar graficas (solo primera partición)
        train_idx, test_idx = next(iter(validador.split(X, y)))
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Gráficas individuales
        graficar_metricas_individuales(
            modelo, X_train, y_train, X_test, y_test,
            nombre_modelo=nombre_modelo, nombre_validacion=nombre_validacion
        )
"""
))

# Celda 8: visualización final
cells.append(nbf.new_code_cell(
    """\
# ================================
# Visualización Comparativa Final
# ================================
plot_metricas_comparativas(resultados)

print("\\n✅ Evaluación y visualización completadas.")
"""
))

# Crear el notebook
notebook = nbf.new_notebook(cells=cells)
notebook_path = Path("notebook/main_pipeline.ipynb")
notebook_path.write_text(nbf.writes(notebook))

notebook_path.name
