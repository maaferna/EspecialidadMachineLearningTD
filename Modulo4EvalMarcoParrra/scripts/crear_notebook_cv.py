from nbformat import v4 as nbf
from pathlib import Path
import nbformat

# Lista de celdas de código, incluyendo instalación de paquetes
code_cells = [
    "# ✅ Instalar dependencias si es necesario",
    "!pip install pandas scikit-learn matplotlib seaborn ray optuna deap",
    "# ✅ Ajustar path para importar desde src/",
    """import sys
from pathlib import Path

project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))""",
    "# ✅ Importar funciones y módulos del proyecto",
    """from src.utils import preprocesar_datos_multiclase_cv
from src.visualizador_cv import visualizar_curva_roc, graficar_metricas_comparativas
from src.optimizador_ray_cv import optimizar_con_raytune_cv
from src.optimizador_genetico_cv import optimizar_con_genetico_cv
from src.modelos import crear_modelo_random_forest, entrenar_modelo_base
from src.evaluador import evaluar_modelo
import pandas as pd""",
    "# ✅ Definir hiperparámetros",
    """NUM_TRIALS = 10
tuned_params = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [5, 10, 15, 20, 30],
    "min_samples_split": [2, 3, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}""",
    "# ✅ Cargar y preprocesar datos",
    """train_df = pd.read_csv("data/Training.csv")
test_df = pd.read_csv("data/Testing.csv")

X_train, y_train, scaler, label_encoder = preprocesar_datos_multiclase_cv(train_df, fit_scaler=True)
X_test, y_test, _, _ = preprocesar_datos_multiclase_cv(test_df, scaler=scaler, label_encoder=label_encoder)

resultados = []""",
    "# ✅ Ray Tune",
    """resultado_ray = optimizar_con_raytune_cv(X_train, y_train, NUM_TRIALS, tuned_params)
resultados.append(resultado_ray)

modelo_ray = crear_modelo_random_forest(**resultado_ray["mejores_parametros"])
modelo_ray.fit(X_train, y_train)
resultado_test_ray = evaluar_modelo("RayTune", modelo_ray, X_test, y_test, 0.0, resultado_ray["mejores_parametros"])
resultados.append(resultado_test_ray)

visualizar_curva_roc(y_test, resultado_test_ray["y_prob"], metodo="RayTune")""",
    "# ✅ Modelo Base",
    """resultado_base = entrenar_modelo_base(X_train, y_train, X_test, y_test)
resultados.append(resultado_base)
visualizar_curva_roc(y_test, resultado_base["y_prob"], metodo="Base")""",
    "# ✅ Genético",
    """resultado_gen = optimizar_con_genetico_cv(X_train, y_train, NUM_TRIALS, tuned_params)
modelo_gen = crear_modelo_random_forest(**resultado_gen["mejores_parametros"])
modelo_gen.fit(X_train, y_train)
resultado_test_gen = evaluar_modelo("Genético", modelo_gen, X_test, y_test, 0.0, resultado_gen["mejores_parametros"])
resultados.append(resultado_test_gen)
visualizar_curva_roc(y_test, resultado_test_gen["y_prob"], metodo="Genético")""",
    "# ✅ Comparar resultados",
    "graficar_metricas_comparativas(resultados)\n\nprint(\"\\n📈 Pipeline completado.\")"
]

# Crear notebook
nb = nbf.new_notebook()
nb.cells = [nbf.new_code_cell(cell) for cell in code_cells]

# Guardar notebook
notebook_path = Path("notebooks") / "Evaluacion_Modelos_Multiclase.ipynb"
notebook_path.parent.mkdir(parents=True, exist_ok=True)
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

notebook_path.name
