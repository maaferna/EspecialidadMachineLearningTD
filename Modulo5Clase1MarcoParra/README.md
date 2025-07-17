# 🧠 Proyecto: Comparación de Técnicas Avanzadas para Predicción de Ingresos

Este proyecto aplica y compara modelos avanzados de **regresión y clasificación** sobre el dataset **Adult Income**, con foco en precisión, estabilidad e interpretabilidad.

## 📁 Estructura del Proyecto

```
.
├── data/                # Dataset 'adult' descargado desde OpenML
├── notebooks/           # Noteb#!/bin/bash

# Crear entorno Conda
echo "📦 Creando entorno conda 'especialidadmachinelearning'..."
conda env create -f environment.yml

# Activar entorno (solo aplica en terminal interactiva)
echo "⚠️  Para activar el entorno, ejecuta manualmente:"
echo "   conda activate especialidadmachinelearning"

# Crear carpetas necesarias
echo "📁 Creando estructura de carpetas..."
mkdir -p data outputs scripts src


echo "✅ Estructura lista. Ejecuta:"
echo "   conda activate especialidadmachinelearning"
echo "   python scripts/main.py"
ook generado al final
├── outputs/             # Imágenes, gráficas, métricas
├── scripts/
│   ├── main.py              # Script principal para entrenamiento y evaluación
│   └── crear_notebook.py    # Script para generar el notebook automáticamente
├── src/
│   ├── utils.py             # Funciones de carga y preprocesamiento
│   ├── modelos.py           # Implementación de ElasticNet, RF, XGBoost, etc.
│   ├── evaluador.py         # Métricas para regresión y clasificación
│   └── visualizador.py      # Visualización: curvas ROC, matrices, comparativas
├── environment.yml          # Entorno conda para reproducibilidad
└── README.md
```

## ⚙️ Modelos Involucrados

- **Elastic Net**
- **Regresión Cuantílica** (percentiles 10, 50, 90)
- **Random Forest**
- **XGBoost**

## 🔍 Evaluación

- Clasificación: accuracy, matriz de confusión, curva ROC
- Regresión: RMSE, Pinball Loss

## 🚀 Instrucciones

1. Crear el entorno:

```bash
conda env create -f environment.yml
conda activate especialidadmachinelearning
```

2. Ejecutar el pipeline:

```bash
python -m scripts.main
```

3. Generar el notebook:

```bash
python -m scripts.crear_notebook
```

## 📌 Notas

- Dataset utilizado: `fetch_openml("adult", version=2)`
- Para regresión cuantílica: usar `QuantileRegressor` de `scikit-learn >=1.1`
- Pipeline modular, orientado a producción y análisis crítico

---


### 🎯 Enfoque Adoptado: Regresión vs Clasificación

El enunciado plantea:

> **“Aplicar y comparar modelos avanzados de regresión y clasificación sobre un mismo problema.”**

A partir de esto, se identifican dos enfoques distintos sobre el mismo dataset (`adult` de OpenML):

* **Clasificación binaria**: Predecir si una persona gana `>50K` o `<=50K` al año.
* **Regresión**: Predecir una **representación continua** o un **percentil estimado** del ingreso (e.g. mediante ElasticNet o Regresión Cuantílica).

#### ✅ Modelos Aplicados

Se utilizaron los siguientes modelos según su naturaleza:

| Modelo               | Tipo de Problema | Implementación                                    |
| -------------------- | ---------------- | ------------------------------------------------- |
| Elastic Net          | Regresión        | `ElasticNet` de `sklearn.linear_model`            |
| Regresión Cuantílica | Regresión        | `QuantileRegressor` con percentiles 0.1, 0.5, 0.9 |
| Random Forest        | Clasificación    | `RandomForestClassifier`                          |
| XGBoost              | Clasificación    | `XGBClassifier`                                   |

#### ❗ Nota Importante

No se aplicaron los modelos de clasificación (Random Forest, XGBoost) a tareas de regresión, ni viceversa, ya que:

* No se indica explícitamente en el enunciado que cada modelo deba ser evaluado en ambos enfoques.
* Cada modelo fue utilizado conforme a su mejor desempeño habitual y su tipo de salida esperada.

---


# 📊 Análisis Detallado de Resultados

## 1. Clasificación: Random Forest y XGBoost

### Rendimiento General

Los modelos de clasificación evaluados fueron **Random Forest** y **XGBoost**. Ambos muestran un desempeño muy competitivo, con **XGBoost ligeramente superior**:

- **XGBoost** obtuvo una precisión (`accuracy`) de **0.8669**.
- **Random Forest** logró **0.8569**.

📌 **Visualización**: gráfico de barras comparativo:  
![Comparación Clasificación y Regresión](outputs/comparacion_metricas_modelos.png)

---

### Curvas ROC y AUC

- **XGBoost** AUC = **0.93**  
![Curva ROC XGBoost](outputs/roc_xgboost.png)

- **Random Forest** AUC = **0.91**  
![Curva ROC RandomForest](outputs/roc_randomforest.png)

---

### Matrices de Confusión

- **XGBoost**  
  - Verdaderos negativos: 6375  
  - Verdaderos positivos: 1466  
  - Falsos negativos: 776  
![Matriz XGBoost](outputs/matriz_confusion_xgboost.png)

- **Random Forest**  
  - Verdaderos negativos: 6434  
  - Verdaderos positivos: 1317  
  - Falsos negativos: 925  
![Matriz RandomForest](outputs/matriz_confusion_randomforest.png)

---

## 2. Regresión: ElasticNet vs Quantile Regression

### Comparación de Métricas

- **ElasticNet** (RMSE): **0.3426**
- **QuantileRegressor** valores de Pinball Loss:
  - QuantileRegressor-0.1: **0.024**
  - QuantileRegressor-0.3: **0.073**
  - QuantileRegressor-0.6: **0.145**

📌 Visualización de métricas por modelo de regresión:  
![Comparación Regresión](outputs/comparacion_regresion_mejores.png)

📌 Comparativa general:  
![Comparativa General](outputs/comparacion_metricas_modelos.png)

---

### Dispersión Cuantil vs Alpha

La gráfica muestra cómo el **Pinball Loss** varía según el cuantil y `alpha`.  
El mejor rendimiento se observa para **α = 0.1** en casi todos los cuantiles.  
![Dispersión Cuantiles](outputs/dispersion_cuantiles.png)

---

## 3. Visualización de Predicción vs Real (ElasticNet)

Se observa que **ElasticNet no logra capturar bien los valores extremos (0 y 1)**, con predicciones concentradas en el centro del rango.

![Predicción vs Real ElasticNet](outputs/pred_vs_real_elasticnet.png)

---

## ✅ Conclusiones Finales

- **XGBoost** es el mejor modelo de clasificación (precisión y AUC).
- **ElasticNet** no es ideal para este problema de regresión.
- **QuantileRegressor**, especialmente con `alpha=0.1`, mostró los mejores resultados de predicción en términos de Pinball Loss.



---

### 📊 Análisis de Resultados – QuantileRegressor con Quantil 0.1

#### 1. **Objetivo del modelo con quantil 0.1**

El modelo `QuantileRegressor` entrenado con un **quantil de 0.1** busca estimar el percentil 10 de la distribución condicional de la variable objetivo. En otras palabras, predice un valor por debajo del cual se espera que se encuentren el 10% de los valores reales, dado un conjunto de variables independientes.

Este tipo de regresión es especialmente útil para entender el **comportamiento de la cola inferior** de la distribución, lo cual puede ser valioso para estudios de **riesgo bajo o rendimiento mínimo**.

---

#### 2. **Resultados observados**

De acuerdo a los resultados entregados en consola:

```
[QuantileRegressor-0.1, α=0.1] Pinball Loss: 0.0244
[QuantileRegressor-0.1, α=0.5] Pinball Loss: 0.0244
[QuantileRegressor-0.1, α=1.0] Pinball Loss: 0.0244
```

* Se observa que el **Pinball Loss es muy bajo (≈0.0244)** y **constante** para todas las combinaciones de `alpha` (0.1, 0.5, 1.0).
* Esto sugiere que **la regularización no tiene un impacto significativo** en la predicción del cuantil 0.1 en este dataset.
* El modelo ha logrado un excelente ajuste para este percentil, lo que se refleja en la gráfica de dispersión, donde los valores predichos se agrupan correctamente en el rango esperado para la cola inferior.

---

#### 3. **Comparación con otros cuantiles**

Comparando el quantil 0.1 con otros como 0.5 (la mediana) o 0.9, observamos que:

* **El Pinball Loss es menor para el quantil 0.1**. Esto puede indicar que:

  * El modelo tiene mayor facilidad para predecir correctamente los valores bajos del target.
  * O bien, que la cola inferior del target es **más estable o menos dispersa**, y por ende más fácil de ajustar.

* En contraste, para cuantiles más altos (por ejemplo 0.7 o 0.9), el error aumenta progresivamente, lo cual podría estar relacionado con **mayor variabilidad en las colas superiores** de la distribución.

---

#### 4. **Implicaciones del resultado**

* **Ventaja operativa**: si el problema de negocio o investigación requiere una predicción conservadora (por ejemplo, una predicción mínima de ingresos, rendimiento o producción), el quantil 0.1 sería una herramienta muy útil.
* **Robustez del modelo**: la consistencia en el Pinball Loss frente a cambios en `alpha` demuestra que el modelo está bien ajustado para este cuantil y **no depende fuertemente de la regularización**.
* **Validación visual**: las gráficas de dispersión confirman que la línea de predicción sigue correctamente la tendencia esperada para valores bajos del objetivo.

---

