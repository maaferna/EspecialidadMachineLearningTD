# crear_notebook.py

import nbformat as nbf
from datetime import datetime

notebook = nbf.v4.new_notebook()
notebook["cells"] = []

# 1. 📦 Instalación de librerías
notebook["cells"].append(nbf.v4.new_code_cell("""\
# ⚙️ Instalación de librerías necesarias
!pip install pandas matplotlib seaborn
"""))

# 2. 🔁 Activar visualización inline
notebook["cells"].append(nbf.v4.new_code_cell("""\
# 📊 Visualizar gráficos en línea
%matplotlib inline
"""))

# 3. 📁 Importar funciones necesarias
notebook["cells"].append(nbf.v4.new_code_cell("""\
# 🔁 Imports del visualizador
from src.visualizador import (
    graficar_accuracy_mejores_modelos,
    graficar_todos_los_resultados,
    graficar_matriz_confusion
)

# Lista de modelos para la matriz de confusión
modelos = ["RandomForest", "AdaBoost", "XGBoost", "LightGBM", "CatBoost"]
"""))

# 4. 📊 Ejecutar visualizaciones
notebook["cells"].append(nbf.v4.new_code_cell("""\
# 📊 Gráfico de accuracies de mejores modelos
graficar_accuracy_mejores_modelos()

# 📊 Gráfico de todos los resultados (todas las combinaciones)
graficar_todos_los_resultados()

# 🔍 Matrices de confusión
for modelo in modelos:
    graficar_matriz_confusion(modelo)

print("✅ Visualizaciones generadas en la carpeta outputs/")
"""))

# Guardar archivo notebook
filename = "notebook_resultados.ipynb"
with open(filename, "w", encoding="utf-8") as f:
    nbf.write(notebook, f)

print(f"📓 Notebook generado: {filename}")
