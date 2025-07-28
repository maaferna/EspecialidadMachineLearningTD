# 🧠 Comparación de Métodos de Boosting y Bagging en Predicción de Ingresos

Este proyecto tiene como objetivo aplicar y comparar técnicas de **ensamblaje supervisado** como **XGBoost**, **AdaBoost** (boosting), y **Random Forest** (bagging), sobre el conjunto de datos real **Adult Income**. La tarea consiste en predecir si una persona gana más de $50K USD anuales en base a información demográfica.

---

## 📁 Estructura del Proyecto

```

├── scripts/
│   ├── main.py               # Script principal que ejecuta todo el pipeline
│   └── build\_notebook.py     # Script para generar notebook interactivo automáticamente
│
├── src/
│   ├── utils.py              # Carga y preprocesamiento del dataset
│   ├── modelos.py            # Entrenamiento de XGBoost, AdaBoost, RandomForest
│   ├── evaluador.py          # Métricas de evaluación
│   ├── visualizador.py       # Visualización de resultados (matriz de confusión, comparativas)
│
├── outputs/                  # Carpeta con CSVs, imágenes y métricas exportadas
│
├── environment.yml           # Ambiente Conda reproducible
├── requirements.txt          # Alternativa con pip
└── README.md                 # Este archivo

````

---

## 🚀 Ejecución del Proyecto

1. **Crear ambiente virtual (opcional):**

```bash
conda env create -f environment.yml
conda activate especialidad-machine-learning
````

2. **Ejecutar script principal:**

```bash
python -m scripts.main
```

---

## 📊 Análisis del Dataset y Preprocesamiento

### 🔹 Dataset: [Adult Income](https://archive.ics.uci.edu/ml/datasets/adult)

* **Origen**: UCI Machine Learning Repository
* **Tamaño original**: 32.561 filas, 15 columnas
* **Tarea**: Clasificación binaria (`income` > 50K USD)

### 🧹 Preprocesamiento Realizado

| Proceso                   | Detalles                                       |
| ------------------------- | ---------------------------------------------- |
| Carga del dataset         | Utiliza `fetch_openml` con versión estándar    |
| Filtrado de valores nulos | Eliminadas 4.262 filas con valores `?`         |
| Binarización del target   | `>50K` → 1, `<=50K` → 0                        |
| Detección de columnas     | 8 categóricas, 6 numéricas                     |
| One-Hot Encoding          | 8 columnas categóricas → 98 columnas dummy     |
| Escalamiento              | Aplicado `StandardScaler` a columnas numéricas |
| División train/test       | 80% entrenamiento, 20% test                    |

### 📦 Dataset final

```text
✅ Dataset limpio con 32.561 filas y 15 columnas.
🧹 Filas con nulos eliminadas: 4.262
🎯 Target: income → binarizado
🔍 Columnas categóricas: 8 → one-hot encoded (98 columnas)
🔍 Columnas numéricas: 6 → escaladas
📊 Total columnas finales: 104

✅ X_train: (24.129, 104)
✅ X_test : (6.033, 104)
```

---

## 🧪 Modelos a Comparar

| Tipo     | Algoritmo     |
| -------- | ------------- |
| Boosting | XGBoost       |
| Boosting | AdaBoost      |
| Bagging  | Random Forest |

### 🔧 Métricas de Evaluación

* Accuracy
* Matriz de Confusión
* Curva ROC / AUC (opcional)
* Visualizaciones comparativas

---

## 📈 Visualizaciones y Resultados

📂 Todos los resultados se almacenan en la carpeta `outputs/`:

| Tipo de salida         | Archivo generado          |
| ---------------------- | ------------------------- |
| Resultados métricas    | `*.csv` por cada modelo   |
| Matriz de confusión    | `confusion_matrix_*.png`  |
| Comparación de modelos | `comparativa_modelos.png` |

---

## 📌 Conclusiones (pendiente)

* Se completará tras entrenar los modelos.
* Se incluirán recomendaciones y elección de modelo más robusto.

---

## 📚 Requisitos

* Python 3.8+
* Scikit-learn
* Pandas / NumPy
* Matplotlib / Seaborn
* XGBoost / LightGBM / CatBoost (opcionales)

---



## 📊 Análisis de Resultados

### 🧠 Resumen Ejecutivo

En este análisis se compararon cinco modelos de clasificación aplicados al conjunto de datos **Adult Income**, con el objetivo de predecir si una persona gana más de 50K anuales en función de variables demográficas.
Se aplicaron técnicas de *Boosting* y *Bagging*, evaluando su rendimiento principalmente con métricas de **accuracy** y **matriz de confusión**.

Los modelos evaluados fueron:

* Random Forest (Bagging)
* AdaBoost (Boosting)
* XGBoost (Boosting)
* LightGBM (Boosting)
* CatBoost (Boosting)

Cada modelo fue ajustado usando **GridSearchCV** y sus mejores resultados fueron almacenados y visualizados.

---

### 📈 Comparación de Accuracy entre los Mejores Modelos

La siguiente figura muestra el accuracy alcanzado por el **mejor conjunto de hiperparámetros** para cada modelo:

![Comparación de Accuracy](outputs/comparacion_accuracy.png)

> ✅ **Interpretación**: El gráfico permite comparar directamente qué modelo fue más preciso. El modelo con mayor barra presentó el mejor desempeño global.

---

### 📦 Distribución de Accuracy por Todas las Configuraciones

Este gráfico muestra la variabilidad del rendimiento (accuracy) para **todas las combinaciones de hiperparámetros** evaluadas en cada modelo:

![Distribución Accuracy Todos los Modelos](outputs/accuracy_todos_modelos.png)

> 🔍 **Análisis**: Este gráfico es útil para ver la **robustez del modelo**. Modelos con distribuciones más compactas y altos valores medianos son más confiables.

---

### 🔍 Matrices de Confusión de los Mejores Modelos

Estas matrices permiten evaluar los errores tipo I y tipo II para cada modelo seleccionado como "mejor":

#### 🔸 CatBoost

![Matriz CatBoost](outputs/confusion_matrix_catboost.png)

#### 🔸 XGBoost

![Matriz XGBoost](outputs/confusion_matrix_xgboost.png)

#### 🔸 Random Forest

![Matriz RandomForest](outputs/confusion_matrix_randomforest.png)

#### 🔸 AdaBoost

![Matriz AdaBoost](outputs/confusion_matrix_adaboost.png)

#### 🔸 LightGBM

![Matriz LightGBM](outputs/confusion_matrix_lightgbm.png)

> 🧮 **Interpretación**: Las diagonales indican las clasificaciones correctas. Los errores (falsos positivos y negativos) se muestran fuera de la diagonal.

---

### 📌 Conclusiones

* El modelo **CatBoost** obtuvo el mejor desempeño en términos de *accuracy*.
* **XGBoost** y **LightGBM** también ofrecieron rendimientos muy competitivos, con distribuciones de accuracy estables.
* **AdaBoost** y **RandomForest** presentaron resultados aceptables, aunque con menor precisión.
* La evaluación exhaustiva mediante múltiples combinaciones de hiperparámetros ayudó a identificar los mejores modelos.
* En términos de interpretación y rendimiento, **CatBoost** sería recomendado para producción en este caso.

---



## ✍️ Autor

Marco Antonio Fernández Parra
Especialización en Machine Learning — Talento Digital Chile

---


