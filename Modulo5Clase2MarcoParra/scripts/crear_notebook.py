import os
import json
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# 1. Crear entorno Conda (opcional si no existe)
conda_setup = """
# ⚙️ Crear entorno Conda
conda create -n especialidad_machine_learning python=3.10 -y
conda activate especialidad_machine_learning
"""

pip_setup = """
# 🐍 Alternativa con pip (si no deseas usar Conda)
pip install pandas matplotlib seaborn scikit-learn statsmodels
"""

# 2. Crear notebook
nb = new_notebook()

nb.cells.append(new_markdown_cell("# 📘 Análisis de Resultados - Especialidad Machine Learning"))
nb.cells.append(new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\n%matplotlib inline"))

# ElasticNet
nb.cells.append(new_markdown_cell("## 🔹 ElasticNet - Resultados"))
nb.cells.append(new_code_cell("""
df_elasticnet = pd.read_csv('outputs/elasticnet_resultados.csv')
df_elasticnet_mejor = pd.read_csv('outputs/elasticnet_mejor.csv')
display(df_elasticnet)
print("\\n🏆 Mejor configuración:")
display(df_elasticnet_mejor)
from IPython.display import Image
Image(filename='outputs/coeficientes_elasticnet.png')
"""))

# Quantile Regression
nb.cells.append(new_markdown_cell("## 🔹 Regresión Cuantílica - Resultados"))
nb.cells.append(new_code_cell("""
df_quantile = pd.read_csv('outputs/quantile_resultados.csv')
df_quantile_best = pd.read_csv('outputs/quantile_mejor.csv')
display(df_quantile)
print("\\n🏆 Mejor cuantil:")
display(df_quantile_best)
from IPython.display import Image
Image(filename='outputs/quantile_pinball_loss.png')
"""))

# VAR
nb.cells.append(new_markdown_cell("## 🔹 VAR - Forecast Macroeconómico"))
nb.cells.append(new_code_cell("""
df_forecast = pd.read_csv('outputs/var_forecast.csv')
display(df_forecast)
from IPython.display import Image
Image(filename='outputs/var_forecast_plot.png')
"""))

# Exportar notebook
with open("build_results_notebook.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("✅ Notebook 'build_results_notebook.ipynb' generado con éxito.")
print("\n--- Instrucciones ---")
print(conda_setup)
print(pip_setup)
