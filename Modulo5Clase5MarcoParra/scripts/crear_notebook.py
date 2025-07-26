from nbformat import v4 as nbf
from pathlib import Path
from nbformat import write

# Crear notebook
nb = nbf.new_notebook()
nb.cells = []

# Celda 1: pip install
nb.cells.append(nbf.new_code_cell("""\
# ✅ Instalación de paquetes necesarios
%pip install pandas matplotlib scikit-learn seaborn --quiet
"""))

# Celda 2: imports
nb.cells.append(nbf.new_code_cell("""\
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

%matplotlib inline
"""))

# Celda 3: carga de datos
nb.cells.append(nbf.new_code_cell("""\
df = pd.read_csv("data/Fish.csv")
df.head()
"""))

# Celda 4: preprocesamiento
nb.cells.append(nbf.new_code_cell("""\
X = df.drop("Weight", axis=1)
y = df["Weight"]
X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
"""))

# Celda 5: definición y entrenamiento de modelos con visualización
nb.cells.append(nbf.new_code_cell("""\
modelos = {
    "lasso": Lasso(alpha=1.0),
    "ridge": Ridge(alpha=0.001),
    "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"📊 {nombre.upper()} - MSE: {mse:.2f}")

    # Gráfico de coeficientes
    coef = modelo.coef_
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, coef)
    plt.title(f"Importancia de Variables - {nombre.capitalize()}")
    plt.xlabel("Coeficiente")
    plt.grid(True)
    plt.show()
"""))

# Celda 6: visualización de resultados guardados
nb.cells.append(nbf.new_code_cell("""\
df_mejores = pd.read_csv("outputs/resultados_gridsearch.csv")
df_todas = pd.read_csv("outputs/todas_las_instancias.csv")

# Gráfico de mejores modelos
plt.figure(figsize=(8,5))
sns.barplot(data=df_mejores, x="modelo", y="mejor_mse")
plt.title("📉 Mejor MSE por Modelo")
plt.ylabel("MSE")
plt.xlabel("Modelo")
plt.grid(True)
plt.show()

# Gráfico de todas las instancias
plt.figure(figsize=(12,6))
sns.lineplot(data=df_todas, x="instancia", y="mse", hue="modelo", marker="o")
plt.title("📊 MSE por Configuración de Parámetros")
plt.xlabel("Instancia evaluada")
plt.ylabel("MSE")
plt.grid(True)
plt.show()
"""))

# Guardar notebook
output_path = Path("notebooks/analisis_regularizacion.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    write(nb, f)



