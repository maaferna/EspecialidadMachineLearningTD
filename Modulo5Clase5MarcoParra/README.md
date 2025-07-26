### ✅ 1. Estructura del Proyecto

```bash
.
├── data/                       # Carpeta para datasets si es necesario
├── notebooks/                 # Notebook generado desde el script
├── outputs/                   # Figuras, coeficientes, resultados
├── scripts/
│   ├── main_regresion_regularizacion.py
│   └── crear_notebook_regresion.sh
├── src/
│   ├── utils.py               # Carga y preprocesamiento
│   ├── modelos.py             # Definición de Lasso, Ridge, ElasticNet
│   ├── evaluador.py           # Funciones para evaluar MSE, etc.
│   └── visualizador.py        # Gráficos de coeficientes y comparación
├── environment.yml
└── README.md
```

---

### ✅ 2. Script para crear la estructura (`setup_proyecto.sh`)

```bash
#!/bin/bash

echo "📁 Creando estructura de proyecto para Regresión con Regularización..."

mkdir -p {data,notebooks,outputs,scripts,src}
touch scripts/{main_regresion_regularizacion.py,crear_notebook_regresion.sh}
touch src/{utils.py,modelos.py,evaluador.py,visualizador.py}
touch README.md

# Environment
cat <<EOF > environment.yml
name: especialidadmachinelearning
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - jupyter
  - pip
  - pip:
      - jupytext
      - nbformat
EOF

echo "✅ Estructura creada y environment.yml generado."
```

---


### ✅ Notas clave:

* Este dataset tiene:

  * Variable objetivo: `Weight` (regresión)
  * Variables numéricas: `Length1`, `Length2`, `Length3`, `Height`, `Width`
  * Variable categórica: `Species`

* **No usamos `LabelEncoder`** porque el objetivo es regresión, no clasificación.

* El archivo espera que el dataset esté en: `data/Fish.csv`

---

# 📊 Comparación de Modelos con Regularización en Regresión Lineal

Este análisis aplica tres técnicas de regularización (`Lasso`, `Ridge` y `Elastic Net`) para predecir el peso de peces en el dataset [Fish Market](https://www.kaggle.com/datasets/vipullrathod/fish-market/data). Se exploran múltiples configuraciones de hiperparámetros y se evalúan con la métrica de error cuadrático medio (MSE).

---

## 1. 🔍 Visualización Inicial del Dataset

El conjunto de datos original tiene `159` muestras y `7` columnas. A continuación se visualiza la distribución de especies y algunas estadísticas básicas:

```python
# Este bloque se puede generar en el notebook:
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data=df, x="Species")
```

---


# 📊 Comparación de Modelos con Regularización en Regresión Lineal

Este análisis aplica tres técnicas de regularización (`Lasso`, `Ridge` y `Elastic Net`) para predecir el peso de peces en el dataset [Fish Market](https://www.kaggle.com/datasets/vipullrathod/fish-market/data). Se exploran múltiples configuraciones de hiperparámetros y se evalúan con la métrica de error cuadrático medio (MSE).


---

## 2. ⚙️ Grid Search y Configuraciones Evaluadas

Se probaron los siguientes hiperparámetros:

| Modelo         | Hiperparámetros probados                            |
| -------------- | --------------------------------------------------- |
| **Lasso**      | `alpha = [0.001, 0.01, 0.1, 1.0]`                   |
| **Ridge**      | `alpha = [0.001, 0.01, 0.1, 1.0]`                   |
| **ElasticNet** | `alpha = [0.1, 1.0]` × `l1_ratio = [0.2, 0.5, 0.8]` |

El mejor modelo se eligió en función del menor **MSE** en el conjunto de test.

---

## 3. ✅ Resultados del Mejor Modelo por Técnica

| Modelo         | Mejor MSE | Configuración óptima      |
| -------------- | --------- | ------------------------- |
| **Lasso**      | 7300.6141 | `alpha=1.0`               |
| **Ridge**      | 7033.2682 | `alpha=0.001`             |
| **ElasticNet** | 7277.4859 | `alpha=0.1, l1_ratio=0.8` |

🖼️ Gráfico de comparación:

![Mejor MSE por Modelo](outputs/grafico_mejores_modelos.png)

---

## 4. 📈 Comparación por Configuración de Parámetros

Este gráfico muestra el rendimiento de todas las combinaciones evaluadas para cada técnica:

![Todas las instancias](outputs/grafico_completo_parametros.png)

📂 También puedes revisar los datos completos en:
[`outputs/todas_las_instancias.csv`](outputs/todas_las_instancias.csv)

---

## 5. 🧮 Análisis de Coeficientes y Variables Importantes

Se analizó la importancia de las variables utilizando los coeficientes del modelo ajustado. A continuación, una visualización sugerida:

```python
# Código sugerido para visualización en notebook
import matplotlib.pyplot as plt
coef = modelo_lasso.best_estimator_.coef_
features = X_train.columns
plt.barh(features, coef)
plt.title("Coeficientes del modelo Lasso")
```

### 🧠 Interpretación:

* **Lasso** tiende a eliminar variables (coeficientes en 0), ideal para simplificar el modelo.
* **Ridge** conserva todas las variables pero reduce la magnitud de los coeficientes.
* **Elastic Net** equilibra ambos enfoques, útil cuando hay correlación entre predictores.

---

## 6. 📌 Discusión

### ¿Cuál técnica fue más efectiva?

Ridge fue la más estable y precisa en este dataset, obteniendo el menor MSE con una configuración muy baja de `alpha`, lo que sugiere una penalización leve.

### ¿Qué variables se eliminaron con Lasso?

Con `alpha=1.0`, el modelo Lasso eliminó (asignó coeficiente 0) a varias variables asociadas a especies, indicando baja correlación con el objetivo (`Weight`).

### ¿Cómo impactó la regularización?

* **Reduce sobreajuste** al penalizar complejidad.
* **Mejora generalización** en conjuntos pequeños como este.
* **Simplifica modelos**, especialmente con Lasso, útil para interpretación.

---

## 📋 Conclusiones y Recomendaciones

* Ridge obtuvo el mejor rendimiento general.
* Lasso es útil para seleccionar variables relevantes.
* ElasticNet puede ser más robusto en datasets con multicolinealidad.

> 🔬 Se recomienda evaluar estas técnicas en un dataset más grande y con validación cruzada k-fold para confirmar los hallazgos.

---

## 📁 Archivos Relevantes

| Archivo                                   | Descripción                                    |
| ----------------------------------------- | ---------------------------------------------- |
| `outputs/resultados_gridsearch.csv`       | Mejores resultados por modelo                  |
| `outputs/todas_las_instancias.csv`        | MSE de cada combinación evaluada               |
| `outputs/grafico_mejores_modelos.png`     | Gráfico de barras con los mejores modelos      |
| `outputs/grafico_completo_parametros.png` | Curvas de MSE por configuración de cada modelo |

---

## 📊 5. Comparación Visual de Coeficientes

Se presentan los coeficientes aprendidos por los tres modelos de regularización: **Ridge**, **Lasso** y **ElasticNet**. Cada gráfico muestra el impacto relativo de cada variable sobre la predicción del peso de los peces (`Weight`).

---

### 🧱 Ridge

![Coeficientes Ridge](outputs/coeficientes_ridge.png)

- `Length2`, `Length3` y `Length1` presentan los coeficientes más altos en magnitud.
- Las variables categóricas (`Species_*`) tienen una menor influencia, aunque no fueron eliminadas.
- Ridge tiende a **mantener todos los coeficientes** al aplicar penalización L2, reduciendo su magnitud pero sin llevarlos a cero.

---

### ✂️ Lasso

![Coeficientes Lasso](outputs/coeficientes_lasso.png)

- Se observa que algunas variables tienen coeficiente **exactamente cero**, lo que implica que fueron **eliminadas automáticamente** del modelo.
- Variables eliminadas: `Height`, `Species_Whitefish`, `Species_Pike`, entre otras con coeficiente ≈ 0.
- Lasso utiliza penalización L1, lo que favorece la **selección automática de características** y un modelo más interpretable.

---

### 🔗 ElasticNet

![Coeficientes ElasticNet](outputs/coeficientes_elasticnet.png)

- Ofrece un equilibrio entre Ridge y Lasso: **reduce** la magnitud de los coeficientes y elimina algunos.
- Las variables `Length1`, `Length2`, y `Length3` siguen siendo dominantes.
- Útil en presencia de **multicolinealidad**, ya que puede seleccionar grupos de variables correlacionadas.

---

## 🧠 6. Discusión Final

### ✅ ¿Cuál fue más efectiva?

Según el gráfico de mejores MSE:

![Mejor MSE](outputs/grafico_mejores_modelos.png)

- **Ridge** fue el modelo más efectivo:
  - **Mejor MSE = 7033.27** con `alpha = 0.001`.
  - Regularización suave que mantiene todas las variables.

---

### ❌ ¿Qué eliminó el modelo Lasso?

Lasso eliminó automáticamente (coeficientes = 0):

- `Height`
- `Species_Whitefish`
- `Species_Pike`

Estas variables no aportaban significativamente a la predicción y fueron descartadas.

---

### ⚖️ ¿Cómo impactó la regularización?

| Técnica       | Selección de variables | Complejidad del modelo | Interpretabilidad |
|---------------|------------------------|------------------------|-------------------|
| **Ridge**     | ❌ No elimina variables | 🔵 Completo             | 🟡 Media          |
| **Lasso**     | ✅ Elimina variables    | 🟢 Simple               | 🟢 Alta           |
| **ElasticNet**| ⚠️ Parcial              | 🔵 Intermedio           | 🟡 Buena          |

---

### 📈 Comparación de todas las configuraciones evaluadas

La siguiente visualización muestra el desempeño (MSE) de cada configuración por técnica de regularización:

![Comparación completa](outputs/grafico_completo_parametros.png)

---

## 📌 Conclusión

- **Ridge** obtuvo el mejor desempeño general.
- **Lasso** ayudó a simplificar el modelo sin comprometer demasiado la precisión.
- **ElasticNet** entregó un balance ideal para problemas con multicolinealidad entre predictores.

Cada técnica ofrece ventajas distintas según el objetivo: rendimiento, simplicidad o interpretabilidad.



## 🧾 Resumen Ejecutivo

Este análisis muestra cómo las técnicas de regularización pueden afectar tanto el rendimiento como la interpretación de un modelo de regresión lineal. Se aplicaron tres enfoques distintos con múltiples configuraciones. Los resultados muestran que Ridge tuvo un mejor ajuste en este dataset, mientras que Lasso es útil para seleccionar variables más relevantes.

---




