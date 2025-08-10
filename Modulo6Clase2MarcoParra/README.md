# 📌 Proyecto: Clustering Basado en Densidad (DBSCAN y HDBSCAN)

## 📖 Resumen Ejecutivo

Este proyecto tiene como objetivo **aplicar, comparar y evaluar algoritmos de clustering basados en densidad** sobre datasets con estructuras complejas y presencia de ruido. Se implementaron **DBSCAN** y **HDBSCAN** para identificar grupos sin necesidad de especificar previamente el número de clusters. El análisis incluye la **reducción de dimensionalidad con PCA**, la **evaluación objetiva mediante métricas de silueta e índice de Davies-Bouldin**, y la **visualización en 2D** para interpretar los resultados. Finalmente, se presenta una **comparación entre ambos algoritmos**, destacando sus fortalezas y limitaciones.

---

## 🎯 Objetivos del Proyecto

1. **Carga de datos**

   * Utilizar datasets adecuados para clustering no supervisado:

     * Generados artificialmente (`make_moons`, `make_blobs`, `make_circles`).
     * Dataset real de vinos (`load_wine` de Scikit-learn).

2. **Preprocesamiento**

   * Escalar los datos con `StandardScaler`.
   * Aplicar PCA en caso de alto número de variables, para simplificar la visualización.

3. **Clustering basado en densidad**

   * Implementar **DBSCAN** con diferentes configuraciones de `eps` y `min_samples`.
   * Implementar **HDBSCAN** sin necesidad de ajuste manual de parámetros.

4. **Evaluación de modelos**

   * Calcular el **índice de silueta** (`silhouette_score`).
   * Calcular el **índice de Davies-Bouldin** (`davies_bouldin_score`).
   * Analizar las diferencias entre ambos algoritmos.

5. **Visualización de resultados**

   * Generar gráficos en 2D mostrando los clusters formados.
   * Comparar cómo DBSCAN y HDBSCAN tratan el ruido y la forma de los clusters.

6. **Conclusiones**

   * Identificar qué algoritmo funcionó mejor en cada caso y por qué.
   * Discutir las limitaciones encontradas y posibles mejoras futuras.

---

## 🧩 Stack Tecnológico

* **Lenguaje**: Python 3.8+
* **Librerías principales**:

  * `scikit-learn` → generación de datasets, escalado, PCA, DBSCAN.
  * `hdbscan` → implementación de HDBSCAN.
  * `matplotlib`, `seaborn` → visualización.
  * `numpy`, `pandas` → manipulación de datos.

---

## 📂 Estructura del Proyecto

```
📁 ClusteringDensidad/
│
├── 📁 scripts/
│   └── main.py                # Pipeline principal
│
├── 📁 src/
│   ├── utils.py               # Carga y preprocesamiento de datasets
│   ├── modelos.py             # Implementación de DBSCAN y HDBSCAN
│   ├── evaluador.py           # Cálculo de métricas de evaluación
│   └── visualizador.py        # Gráficas comparativas de clusters
│
├── 📁 outputs/                # Resultados generados
│   ├── clusters_dbscan.png
│   ├── clusters_hdbscan.png
│   ├── comparacion_metricas.csv
│
├── environment.yml            # Entorno Conda con dependencias
└── create_notebook.py         # Script para generar notebook con resultados
```


---


## 📊 Resultados – DBSCAN

### 1️⃣ Introducción al experimento

El algoritmo **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** se probó sobre el dataset *Wine* preprocesado, aplicando escalado **MinMaxScaler** y reduciendo dimensionalidad para visualización en dos ejes (`alcohol` y `malic_acid`).

Se evaluaron diferentes combinaciones de parámetros clave:

* **`eps`**: distancia máxima para considerar puntos como vecinos.
* **`min_samples`**: número mínimo de vecinos requeridos para formar un núcleo de clúster.

### 2️⃣ Observaciones generales

En todas las combinaciones probadas, **el algoritmo detectó un único clúster (etiqueta 0)**, sin diferenciar subgrupos. Esto indica que:

* Con las distancias y densidades actuales, DBSCAN no encuentra regiones con densidad lo suficientemente distinta como para separar en más clústeres.
* Los valores de `eps` seleccionados (0.3, 0.5, 0.7) y `min_samples` (3, 5, 10) no producen cambios significativos en la estructura de agrupamiento.

### 3️⃣ Análisis por configuración

#### 🔹 `eps = 0.3`

* **min\_samples = 3, 5, 10** → No hay separación clara de grupos, todos los puntos pertenecen al mismo clúster.
* Un valor tan bajo de `eps` limita el alcance de cada punto, pero en este dataset las distancias no generan densidad suficiente para separar grupos.

#### 🔹 `eps = 0.5`

* **min\_samples = 3, 5, 10** → El resultado es idéntico al caso anterior: un único clúster.
* El ligero aumento de `eps` tampoco es suficiente para abarcar más puntos y formar subgrupos.

#### 🔹 `eps = 0.7`

* **min\_samples = 3, 5, 10** → Se mantiene la situación: un único clúster.
* Aunque el radio de búsqueda es mayor, la distribución de datos sigue siendo homogénea en términos de densidad, impidiendo la detección de fronteras naturales.

### 4️⃣ Interpretación

El comportamiento observado sugiere que:

* El dataset *Wine*, en las dimensiones seleccionadas (`alcohol` y `malic_acid`), **no presenta densidades locales suficientemente contrastadas** para que DBSCAN identifique más de un grupo.
* Este resultado pone en evidencia la **sensibilidad de DBSCAN a la escala de los datos y a la selección de `eps`**, así como la necesidad de evaluar más dimensiones (por ejemplo, aplicando PCA antes de DBSCAN) para descubrir patrones más complejos.


---


## 📊 Resultados y Análisis Comparativo

### Objetivo Recordatorio

El objetivo de este proyecto fue **aplicar, comparar y evaluar modelos de clustering basados en densidad** —DBSCAN y HDBSCAN— sobre el dataset **Wine** (preprocesado con escalado y reducción de dimensionalidad mediante PCA), evaluando el rendimiento con métricas objetivas (Índice de Silueta y Davies-Bouldin) y mediante visualizaciones en 2D/3D, incluyendo PCA y t-SNE.

---

### 1️⃣ Resultados de DBSCAN

Se evaluaron múltiples combinaciones de parámetros `eps` y `min_samples` para DBSCAN, con el fin de observar la sensibilidad del algoritmo a estos hiperparámetros.

#### 📌 Observaciones clave:

* En **todas las configuraciones** (ver Figuras [`dbscan_eps*.png`](outputs/)), DBSCAN tendió a asignar todos los puntos a un único cluster (etiqueta 0), sin identificar subgrupos claros.
* Esto sugiere que, con la estructura actual de los datos (13 features reducidos por PCA a 2D para visualización), las distancias entre puntos no son lo suficientemente diferentes como para que DBSCAN detecte zonas de densidad distintas.
* El t-SNE aplicado sobre los resultados de DBSCAN (ver Figuras [`tsne_dbscan_eps*.png`](outputs/)) confirmó que el espacio embebido no mostró agrupaciones claras, evidenciando que DBSCAN no logró separar patrones relevantes en este dataset sin un ajuste más agresivo de parámetros.

 

#### 📊 Tablas asociadas:

* [**DBSCAN\_consolidado.csv**](outputs/DBSCAN_consolidado.csv): contiene el índice de silueta y Davies-Bouldin para cada combinación de parámetros, confirmando valores de silueta cercanos a 0 y Davies-Bouldin altos, lo que refleja baja calidad de clustering.

---

### 2️⃣ Resultados de HDBSCAN

Se probó HDBSCAN con diferentes valores de `min_cluster_size` (5 y 10).

#### 📌 Observaciones clave:

* A diferencia de DBSCAN, **HDBSCAN sí identificó múltiples clusters** en ambas configuraciones (ver Figuras [`hdbscan_mincluster*.png`](outputs/)).
* Detectó entre 3 y 4 clusters principales, junto con algunos puntos clasificados como ruido (etiqueta `-1`).
* Esto es consistente con la naturaleza jerárquica de HDBSCAN, que permite detectar clusters de distinta densidad sin necesidad de definir `eps` manualmente.
* El t-SNE aplicado a HDBSCAN (ver Figuras [`tsne_hdbscan_mincluster*.png`](outputs/)) mostró una mejor separación entre grupos en el espacio reducido, reforzando la idea de que HDBSCAN modeló mejor la estructura latente.

#### 📊 Tablas asociadas:

* [**HDBSCAN\_consolidado.csv**](outputs/HDBSCAN_consolidado.csv): refleja índices de silueta más altos y Davies-Bouldin más bajos que DBSCAN, indicando una mejor calidad de agrupamiento.

---

### 3️⃣ Comparativa PCA vs. Clustering Basado en Densidad

El PCA 2D del dataset (Figura [`pca_2d.png`](outputs/pca_2d.png)) muestra claramente que existen **tres grupos naturales** en el espacio reducido, que corresponden a las clases originales del dataset Wine.

* **DBSCAN** no logró identificar estos tres grupos, probablemente porque la métrica de distancia en el espacio completo de features no reflejaba la estructura que sí es visible tras PCA.
* **HDBSCAN** se acercó más a la estructura esperada, detectando subgrupos que se alinean parcialmente con los grupos del PCA.

---

### 4️⃣ Conclusiones

1. **DBSCAN** fue muy sensible a la elección de parámetros y, en este caso, no logró segmentar los datos en clusters significativos, resultando en una única clase para la mayoría de configuraciones.
2. **HDBSCAN** mostró mayor robustez, detectando varios clusters sin necesidad de ajustar manualmente `eps` y con mejor alineación a la estructura real del dataset.
3. La visualización con **PCA** y **t-SNE** evidenció que los datos tienen una estructura subyacente que HDBSCAN captura mejor que DBSCAN.
4. Para datasets con ruido y clusters de distinta densidad, **HDBSCAN** es preferible a DBSCAN, especialmente cuando no se conoce un `eps` óptimo.
5. En un flujo DSNA (Data Science for Non-Analysts), este tipo de análisis comparativo ayuda a seleccionar un algoritmo de clustering más interpretable y efectivo.

---
# Análisis de Clustering en Dataset Wine 📊

Este repositorio contiene los resultados del análisis de clustering utilizando **DBSCAN** y **HDBSCAN** en el dataset Wine.  
Se evaluaron distintas configuraciones de parámetros y se visualizaron los resultados mediante **PCA 2D** y **t-SNE**.

---

## Resultados - DBSCAN 🎯

### Configuraciones Evaluadas
- eps = 0.3 con min_samples = 3, 5, 10
- eps = 0.5 con min_samples = 3, 5, 10
- eps = 0.7 con min_samples = 3, 5, 10

### Visualizaciones DBSCAN

1. **eps=0.3, min_samples=3**  
   ![DBSCAN eps=0.3 min=3](outputs/dbscan_eps0.3_min3.png)

2. **eps=0.3, min_samples=5**  
   ![DBSCAN eps=0.3 min=5](outputs/dbscan_eps0.3_min5.png)

3. **eps=0.3, min_samples=10**  
   ![DBSCAN eps=0.3 min=10](outputs/dbscan_eps0.3_min10.png)

4. **eps=0.5, min_samples=3**  
   ![DBSCAN eps=0.5 min=3](outputs/dbscan_eps0.5_min3.png)

5. **eps=0.5, min_samples=5**  
   ![DBSCAN eps=0.5 min=5](outputs/dbscan_eps0.5_min5.png)

6. **eps=0.5, min_samples=10**  
   ![DBSCAN eps=0.5 min=10](outputs/dbscan_eps0.5_min10.png)

7. **eps=0.7, min_samples=3**  
   ![DBSCAN eps=0.7 min=3](outputs/dbscan_eps0.7_min3.png)

8. **eps=0.7, min_samples=5**  
   ![DBSCAN eps=0.7 min=5](outputs/dbscan_eps0.7_min5.png)

9. **eps=0.7, min_samples=10**  
   ![DBSCAN eps=0.7 min=10](outputs/dbscan_eps0.7_min10.png)

---

## Resultados - HDBSCAN 🌳

### Configuraciones Evaluadas
- min_cluster_size = 5
- min_cluster_size = 10

### Visualizaciones HDBSCAN

1. **min_cluster_size=5**  
   ![HDBSCAN min_cluster_size=5](outputs/hdbscan_mincluster5.png)

2. **min_cluster_size=10**  
   ![HDBSCAN min_cluster_size=10](outputs/hdbscan_mincluster10.png)

---

## Visualizaciones t-SNE 🧩

### t-SNE DBSCAN
- eps=0.3, min_samples=3 → ![t-SNE DBSCAN eps=0.3 min=3](outputs/tsne_dbscan_eps0.3_min3.png)  
- eps=0.3, min_samples=5 → ![t-SNE DBSCAN eps=0.3 min=5](outputs/tsne_dbscan_eps0.3_min5.png)  
- eps=0.3, min_samples=10 → ![t-SNE DBSCAN eps=0.3 min=10](outputs/tsne_dbscan_eps0.3_min10.png)  
- eps=0.5, min_samples=3 → ![t-SNE DBSCAN eps=0.5 min=3](outputs/tsne_dbscan_eps0.5_min3.png)  
- eps=0.5, min_samples=5 → ![t-SNE DBSCAN eps=0.5 min=5](outputs/tsne_dbscan_eps0.5_min5.png)  
- eps=0.5, min_samples=10 → ![t-SNE DBSCAN eps=0.5 min=10](outputs/tsne_dbscan_eps0.5_min10.png)  
- eps=0.7, min_samples=3 → ![t-SNE DBSCAN eps=0.7 min=3](outputs/tsne_dbscan_eps0.7_min3.png)  
- eps=0.7, min_samples=5 → ![t-SNE DBSCAN eps=0.7 min=5](outputs/tsne_dbscan_eps0.7_min5.png)  
- eps=0.7, min_samples=10 → ![t-SNE DBSCAN eps=0.7 min=10](outputs/tsne_dbscan_eps0.7_min10.png)  

### t-SNE HDBSCAN
- min_cluster_size=5 → ![t-SNE HDBSCAN min_cluster_size=5](outputs/tsne_hdbscan_mincluster5.png)  
- min_cluster_size=10 → ![t-SNE HDBSCAN min_cluster_size=10](outputs/tsne_hdbscan_mincluster10.png)  

---

## PCA 2D 📉
![PCA 2D Visualization](outputs/pca_2d.png)

---

## Datos Consolidados 📂
- [DBSCAN_consolidado.csv](outputs/DBSCAN_consolidado.csv) → Métricas para todas las configuraciones DBSCAN  
- [HDBSCAN_consolidado.csv](outputs/HDBSCAN_consolidado.csv) → Métricas para configuraciones HDBSCAN



## 📝 Conclusiones

* **DBSCAN** es muy útil para detectar clusters de forma arbitraria, pero su rendimiento depende fuertemente de los parámetros `eps` y `min_samples`.
* **HDBSCAN** ofrece mayor robustez al no requerir ajuste manual, adaptándose mejor a datasets con ruido o estructuras complejas.
* El **índice de silueta** y el **índice de Davies-Bouldin** permitieron validar objetivamente las agrupaciones, mostrando que **HDBSCAN suele generar clusters más naturales y estables**.
* Este estudio demuestra la importancia de combinar **métricas objetivas** con **visualizaciones exploratorias** para interpretar adecuadamente los resultados de clustering no supervisado.
