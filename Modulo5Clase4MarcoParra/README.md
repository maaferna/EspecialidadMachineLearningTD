
# Proyecto: Aplicación de Técnicas de Validación Cruzada

## 🎯 Objetivo

Aplicar y comparar diferentes técnicas de validación cruzada y métricas de evaluación sobre un modelo predictivo, utilizando datos reales. El objetivo principal es interpretar y justificar los resultados obtenidos en base a métricas de rendimiento como:

- Precisión
- Recall
- F1-Score
- Matriz de Confusión
- Curvas ROC y Precision-Recall

---

## 🧠 Contexto del Problema

Se utiliza el dataset **Adult Census Income** (descargado desde [OpenML](https://www.openml.org/d/1590)) para simular un caso en que se desea predecir si un cliente abandonará un servicio. Para ello, se binariza la variable objetivo `class`:

- `>50K` → `1.0` (simula "abandona")
- `<=50K` → `0.0` (simula "permanece")

---

## 🧹 Preprocesamiento

Se realiza un preprocesamiento modular sobre el dataset:

- Eliminación de valores faltantes (`NaN`)
- Conversión de la variable objetivo a tipo binario (`float`)
- Codificación de variables categóricas mediante One-Hot Encoding
- Estandarización de variables numéricas con `StandardScaler`

> ⚠️ No se aplica `train_test_split` ya que se evaluará completamente con técnicas de validación cruzada.

---

## 🔁 Técnicas de Validación Cruzada Aplicadas

El proyecto contempla múltiples técnicas, ajustadas al contexto del dataset:

- **K-Fold Cross Validation**
- **Leave-One-Out (LOO)**
- **Stratified K-Fold** (para mantener proporciones en datasets desbalanceados)
- *(Time Series CV descartado, pues el dataset no tiene estructura temporal)*

---

## 📊 Modelos Utilizados

- **Random Forest**
- **Logistic Regression**

Estos modelos serán evaluados mediante todas las técnicas de validación cruzada descritas.

---

## 📈 Métricas de Evaluación

Se reportarán y compararán las siguientes métricas:

- Accuracy
- Recall
- F1-Score
- Matriz de Confusión
- ROC AUC
- Precision-Recall AUC

---

## 📦 Estructura del Proyecto

```

validacion\_cruzada/
│
├── src/
│   ├── utils.py           # Carga y preprocesamiento
│   ├── validacion.py         # Funciones para validación cruzada
│   ├── entrenamiento.py      # Entrenamiento y evaluación de modelos
│   ├── visualizacion.py      # Gráficos: matriz, curvas, barras
│
├── main.py                   # Ejecuta todo el pipeline
├── outputs/                  # Carpeta donde se guardan las figuras
└── README.md                 # Este archivo

````

---

## 🛠️ Requisitos

- Python 3.8+
- scikit-learn
- pandas
- seaborn
- matplotlib

Instalación recomendada:

```bash
conda activate especialidadmachinelearning
````

---

## ✅ Estado Actual

* [x] Dataset cargado y preprocesado
* [x] Objetivo binarizado correctamente
* [x] Eliminado `train_test_split` (se usará validación cruzada)
* [ ] Validación cruzada implementada (próximo paso)
* [ ] Visualizaciones agregadas
* [ ] Análisis de resultados

---

## 📬 Autores

Trabajo desarrollado en modalidad grupal como parte de la especialización en Machine Learning.
Tiempo estimado de ejecución: 120 minutos.


---

## 📊 Resultados y Evaluación Comparativa de Modelos

Se evaluaron los modelos **RandomForest** y **LogisticRegression** utilizando dos estrategias de validación cruzada: `KFold` y `StratifiedKFold`. Para cada combinación se calcularon métricas de desempeño y se generaron gráficas de evaluación detalladas.

### 🔍 Comparación General de Métricas

La figura a continuación muestra un resumen de las métricas (`accuracy`, `precision`, `recall`, `f1_score`) promediadas por cada estrategia:

![Comparación de Métricas](outputs/comparacion_metricas_validacion.png)

Se observa un rendimiento muy similar entre ambas técnicas, con ligeras ventajas en `f1_score` y `recall` usando `StratifiedKFold`, que conserva mejor la proporción entre clases durante la partición.

---

### 🔎 Matrices de Confusión

Las siguientes gráficas muestran la distribución de verdaderos positivos, negativos, falsos positivos y falsos negativos para cada modelo:

* **LogisticRegression + KFold**
  ![Matriz Confusión](outputs/matriz_confusion_LogisticRegression_KFold.png)
* **LogisticRegression + StratifiedKFold**
  ![Matriz Confusión](outputs/matriz_confusion_LogisticRegression_StratifiedKFold.png)
* **RandomForest + KFold**
  ![Matriz Confusión](outputs/matriz_confusion_RandomForest_KFold.png)
* **RandomForest + StratifiedKFold**
  ![Matriz Confusión](outputs/matriz_confusion_RandomForest_StratifiedKFold.png)

---

### 📈 Curvas Precision-Recall

Estas curvas permiten analizar la relación entre `precision` y `recall` para distintos umbrales de decisión:

* LogisticRegression + KFold
  ![PR-Curve](outputs/precision_recall_LogisticRegression_KFold.png)
* LogisticRegression + StratifiedKFold
  ![PR-Curve](outputs/precision_recall_LogisticRegression_StratifiedKFold.png)
* RandomForest + KFold
  ![PR-Curve](outputs/precision_recall_RandomForest_KFold.png)
* RandomForest + StratifiedKFold
  ![PR-Curve](outputs/precision_recall_RandomForest_StratifiedKFold.png)

---

## ⚠️ Exclusión de Leave-One-Out (LOOCV)

Inicialmente se consideró la estrategia `LeaveOneOut`, pero fue descartada en etapas tempranas debido a:

* **Costos computacionales excesivos**: el dataset tiene más de 48.000 muestras. LOOCV genera **una iteración por cada observación**, lo que resulta en más de 48.000 ciclos de entrenamiento y evaluación.
* **Riesgo de overfitting**: LOOCV puede tener alta varianza en datasets grandes y desbalanceados, generando métricas inconsistentes.
* **Resultados no representativos**: en pruebas iniciales, muchas métricas resultaban nulas o indefinidas (ver advertencias `UndefinedMetricWarning`), especialmente cuando los folds contenían solo ejemplos de una clase.

Por estos motivos, **se optó por estrategias más eficientes y robustas como `KFold` y `StratifiedKFold`**, que permitieron conservar balance y reducir los tiempos de ejecución drásticamente.

---

Aquí tienes una sección redactada en **Markdown** para tu informe final, que incluye:

* Resultados generales
* Justificación de la exclusión de Leave-One-Out (LOOCV)
* Referencias a las imágenes generadas.


---

### 🧪 Curvas ROC (Receiver Operating Characteristic)

Las curvas ROC muestran la tasa de verdaderos positivos contra la tasa de falsos positivos. En este análisis, todos los modelos alcanzaron un **AUC (Area Under Curve) de aproximadamente 0.90**, lo que indica un buen rendimiento.

#### RandomForest

![ROC KFold](outputs/roc_curve_RandomForest_KFold.png)
![ROC StratifiedKFold]outputs/(roc_curve_RandomForest_StratifiedKFold.png)

#### LogisticRegression

![ROC KFold](outputs/roc_curve_LogisticRegression_KFold.png)
![ROC StratifiedKFold](outputs/roc_curve_LogisticRegression_StratifiedKFold.png)

---

## ✅ Conclusiones

* **StratifiedKFold** fue ligeramente más robusto que KFold en términos de métricas debido al manejo equilibrado de clases.
* **RandomForest** y **LogisticRegression** presentaron rendimientos muy similares, aunque RandomForest tuvo una ligera ventaja en `recall` y `f1_score`.
* La exclusión de **LOOCV** fue una decisión justificada por razones de eficiencia computacional y confiabilidad de métricas.

