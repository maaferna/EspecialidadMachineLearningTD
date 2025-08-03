# 📊 Proyecto de Clustering Jerárquico con PCA y t-SNE

## 📌 Introducción

Este proyecto tiene como objetivo aplicar **técnicas de clustering jerárquico aglomerativo** en datasets multivariados (Iris y Wine) para identificar estructuras ocultas en los datos.
Se utilizarán métodos de reducción de dimensionalidad (**PCA** y **t-SNE**) con el fin de visualizar mejor los patrones y evaluar la calidad de los agrupamientos generados.

La idea central es **explorar la similitud entre observaciones sin necesidad de etiquetas supervisadas** y posteriormente analizar los resultados mediante gráficos y métricas descriptivas.

---

## 🎯 Objetivos del proyecto

1. **Preprocesar los datos** (normalización, eliminación de outliers, revisión de varianza).
2. **Aplicar Clustering Jerárquico Aglomerativo** con diferentes configuraciones:

   * Método **Ward** (minimiza la varianza intra-cluster).
   * Método **Average** (utiliza la distancia promedio entre elementos).
3. **Generar dendrogramas** para visualizar la estructura jerárquica de los clusters.
4. **Comparar resultados con PCA y t-SNE** en 2D y 3D.
5. **Analizar las diferencias** entre la proyección PCA y t-SNE en la identificación de grupos.
6. Documentar los resultados en gráficos y tablas exportadas a la carpeta `outputs/`.

---

## 🛠️ Tecnologías utilizadas

* **Lenguaje principal:** Python 3.8
* **Librerías de Machine Learning y análisis**:

  * `scikit-learn` → clustering, PCA, t-SNE
  * `scipy` → linkage y dendrogramas
  * `numpy` y `pandas` → manipulación de datos
* **Visualización**:

  * `matplotlib` → gráficos 2D y 3D
  * `seaborn` → visualizaciones estadísticas
* **Entorno**: Conda (`environment.yml` con dependencias)
* **Otros**: `openml` para cargar datasets públicos

---

## 📂 Estructura de carpetas

```
├── scripts/
│   └── main.py              # Archivo principal del pipeline
├── src/
│   ├── utils.py             # Métodos para carga y preprocesamiento de datos
│   ├── modelos.py           # Definición de modelos de clustering
│   ├── evaluador.py         # Funciones para evaluar PCA y clustering
│   └── visualizador.py      # Gráficas (PCA, t-SNE, dendrogramas)
├── outputs/                 # Resultados exportados (gráficas, CSVs)
├── notebooks/
│   └── analisis.ipynb       # Notebook con resultados y visualizaciones
├── environment.yml          # Dependencias Conda del proyecto
└── README.md                # Documentación del proyecto
```

---

## ⚙️ Flujo de trabajo del proyecto

1. **Carga de dataset** desde `utils.py`

   * Dataset elegido: Iris o Wine desde `sklearn.datasets`.
   * Escalado con `StandardScaler` o `MinMaxScaler`.

2. **Reducción de dimensionalidad (PCA y t-SNE)**

   * PCA → cálculo de varianza acumulada y proyecciones.
   * t-SNE → proyección no lineal para comparación.

3. **Clustering jerárquico aglomerativo**

   * Configuración con métodos **Ward** y **Average**.
   * Evaluación de 2 y 3 clusters.
   * Visualización con dendrogramas y proyecciones.

4. **Exportación de resultados**

   * CSV con métricas de PCA.
   * Gráficas de PCA y t-SNE en 2D y 3D.
   * Dendrogramas comparativos.

---


# 📊 Informe de Resultados – Clustering Jerárquico con PCA y t-SNE

## 📝 Introducción

El objetivo de este proyecto fue aplicar **clustering jerárquico aglomerativo** sobre datasets multivariados (ejemplo: *Iris*), utilizando métodos de enlace **Ward** y **Average**, y posteriormente visualizar y analizar la estructura de los grupos formados.

Se emplearon dos técnicas de reducción de dimensionalidad para facilitar la interpretación:

1. **PCA (Análisis de Componentes Principales)** → Busca proyectar los datos en menos dimensiones **maximizando la varianza explicada**.
2. **t-SNE (t-distributed Stochastic Neighbor Embedding)** → Busca proyectar los datos en un espacio de baja dimensión **preservando la estructura local** (distancias entre puntos vecinos).

Además, se generaron **dendrogramas** para visualizar la jerarquía de los clusters formados.

---

## 📍 Análisis de Gráficos

### 🔹 1. PCA en 2D

![PCA 2D](outputs/pca_2d.png)

En este gráfico, los datos fueron reducidos a **2 componentes principales**.

**Interpretación:**

* Los colores representan las clases reales o clusters detectados.
* Se observa que los puntos se agrupan formando **tres regiones principales**:

  * El grupo de la izquierda se encuentra bien separado.
  * Los dos grupos de la derecha tienen una ligera superposición, lo que indica que PCA conserva bastante información, pero no logra una separación perfecta.
* El **eje X (Componente Principal 1)** captura la mayor varianza, seguido del **eje Y (Componente Principal 2)**.

📌 **Conclusión parcial:** PCA 2D permite una visualización clara de la estructura general, especialmente destacando un cluster muy definido.

---

### 🔹 2. PCA en 3D

![PCA 3D](outputs/pca_3d.png)

Aquí los datos se proyectaron en **3 componentes principales**.

**Interpretación:**

* La tercera dimensión ayuda a **mejorar la separación** de los clusters que en 2D se solapaban parcialmente.
* Se distinguen **tres agrupamientos claros**, reforzando que la estructura intrínseca de los datos efectivamente corresponde a 3 grupos.
* El PCA sigue siendo lineal, por lo que no capta relaciones altamente no lineales, pero es suficiente para este dataset.

📌 **Conclusión parcial:** PCA 3D confirma la evidencia visual de tres clusters claros, mejorando la discriminación respecto al PCA 2D.

---

### 🔹 3. Dendrograma – Average (3 clusters)

![Dendrograma Average 3](outputs/dendrograma_average_3.png)

Un **dendrograma** muestra la jerarquía de fusiones de los clusters.

**Interpretación:**

* El eje Y indica la **distancia de enlace** (qué tan similares son los grupos antes de fusionarse).
* Con el método **Average**, cada fusión se calcula en base al **promedio de distancias entre todos los puntos de dos clusters**.
* Se observa un **salto grande en la altura** cuando se pasa de 3 a 2 clusters, lo que sugiere que **3 clusters es una partición natural**.

📌 **Conclusión parcial:** Este dendrograma respalda la elección de 3 clusters, ya que cortar el árbol a esa altura preserva la estructura natural.

---

### 🔹 4. t-SNE – Average (3 clusters)

![t-SNE Average 3](outputs/tsne_average_3.png)

**Interpretación:**

* A diferencia de PCA, t-SNE no busca maximizar varianza, sino preservar **distancias locales**.
* Los tres clusters aparecen **muy bien definidos** y separados, con **menor solapamiento** que en PCA 2D.
* Esto indica que t-SNE puede capturar **estructuras no lineales** que PCA no logra visualizar.

📌 **Conclusión parcial:** t-SNE es más efectivo que PCA 2D para mostrar separaciones claras entre clusters.

---

### 🔹 5. Comparación de 2 y 3 Clusters (Ward y Average)

* **Average con 2 clusters:**
  ![t-SNE Average 2](outputs/tsne_average_2.png)
  ![Dendrograma Average 2](outputs/dendrograma_average_2.png)
  ➝ Los datos se dividen en dos grandes grupos, pero la separación pierde detalle respecto a 3 clusters.

* **Ward con 3 clusters:**
  ![t-SNE Ward 3](outputs/tsne_ward_3.png)
  ![Dendrograma Ward 3](outputs/dendrograma_ward_3.png)
  ➝ El método **Ward** minimiza la varianza interna de cada cluster, lo que genera **clusters más compactos y equilibrados**.

* **Ward con 2 clusters:**
  ![t-SNE Ward 2](outputs/tsne_ward_2.png)
  ![Dendrograma Ward 2](outputs/dendrograma_ward_2.png)
  ➝ Similar a Average 2, pero con una estructura más balanceada en tamaño.

📌 **Conclusión comparativa:**

* **Ward** tiende a producir clusters más equilibrados y compactos.
* **Average** refleja más la densidad relativa entre grupos.
* Con **2 clusters**, se simplifica demasiado la estructura.
* Con **3 clusters**, se obtiene la mejor representación de la variabilidad de los datos.

---

## 📌 Diferencias clave entre PCA y t-SNE

| Aspecto                | PCA                                 | t-SNE                                   |
| ---------------------- | ----------------------------------- | --------------------------------------- |
| Tipo de reducción      | Lineal (proyecciones ortogonales)   | No lineal (preserva distancias locales) |
| Interpretabilidad      | Alta (basada en varianza explicada) | Media (parámetros sensibles, no lineal) |
| Separación de clusters | Buena, pero puede solaparse         | Excelente para visualizar separaciones  |
| Criterio de selección  | Varianza explicada acumulada        | Distribución visual de agrupamientos    |

---

## 📊 Resumen Ejecutivo

* **El PCA 2D y 3D confirmaron que la estructura natural del dataset es de 3 clusters**, aunque el PCA 2D mostró cierto solapamiento que se redujo en PCA 3D.
* **t-SNE demostró una separación más clara y definida**, mostrando la capacidad de este método para capturar relaciones no lineales.
* **Los dendrogramas evidenciaron un salto en la distancia de enlace al pasar de 3 a 2 clusters**, validando que 3 es la mejor partición.
* El método **Ward produjo clusters más compactos**, mientras que **Average reflejó mejor las densidades relativas**.
* **Conclusión final:** Para este dataset, la combinación de PCA (como exploración inicial) y t-SNE (para separación más clara) con **3 clusters** ofrece la representación más coherente de la estructura de los datos.

---



## 📑 Resumen Ejecutivo

El presente proyecto implementa un pipeline para analizar y visualizar **estructuras jerárquicas de clusters** en datasets multivariados.
A través del uso de **PCA y t-SNE**, se explora cómo distintas técnicas de reducción de dimensionalidad impactan la representación visual de los datos y la claridad de los agrupamientos.

Los principales hallazgos esperados son:

* Identificación de **agrupamientos naturales** en Iris y Wine.
* Confirmación de que **Ward** suele producir clusters más compactos, mientras que **Average** puede resaltar relaciones más difusas.
* Demostración de que PCA y t-SNE ofrecen **perspectivas complementarias**:

  * PCA conserva la varianza global.
  * t-SNE enfatiza la **estructura local** de los datos.

La comparación de ambos métodos permitirá **evaluar fortalezas y limitaciones** en la detección de patrones y justificar cuál sería más recomendable para problemas de clasificación sin supervisión en entornos reales.

---


