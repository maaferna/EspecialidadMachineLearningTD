# 📊 Resultados: Reducción de Dimensionalidad con PCA y KNN

Este proyecto explora la reducción de dimensionalidad mediante **Análisis de Componentes Principales (PCA)** y su aplicación como preprocesamiento para **K-Nearest Neighbors (KNN)**, utilizando el dataset *Iris*.  

---

## 📌 Resumen Ejecutivo

- **PCA no supervisado** permitió reducir de 4 a **2 componentes principales**, conservando el **95.7% de la varianza**.
- Se observó que con **2 componentes** se logra una clara separación entre las clases, facilitando la interpretación y reduciendo la dimensionalidad sin perder información significativa.
- Al aplicar **KNN sobre el espacio PCA**, se alcanzó un desempeño sobresaliente, con una **Accuracy máxima de 100% con K=7**.
- Los resultados muestran que **PCA es una herramienta efectiva** para simplificar el espacio de características y que **KNN mantiene o mejora su rendimiento tras la reducción**.

---

## 📌 Varianza Explicada Acumulada

La siguiente tabla muestra la varianza acumulada para diferentes números de componentes:

| N° Componentes | Varianza Explicada |
|----------------|--------------------|
| 2              | 0.957              |
| 3              | 0.993              |
| 4              | 1.000              |

📉 El umbral recomendado de **95%** se alcanza con **2 componentes**.

![Varianza Explicada](outputs/pca_varianza_explicada.png)

---

## 📌 Visualización PCA

### Proyección en 2D con las Clases Reales  
La reducción a 2 componentes permitió visualizar con claridad la separación de las tres cluster del dataset *Iris*.

![PCA en 2D](outputs/pca_2d.png)

### Proyección en 3D con KMeans  
Se observa que los agrupamientos coinciden con las clases originales, validando la calidad de la reducción dimensional.

![PCA en 3D](outputs/pca_cluster_3d.png)

---

## 📌 Evaluación de KNN con PCA Óptimo

Tras aplicar PCA con **2 componentes**, se evaluó KNN con diferentes valores de K.

| K Vecinos | Accuracy | F1-Score |
|-----------|----------|----------|
| 3         | 0.933    | 0.933    |
| 5         | 0.967    | 0.967    |
| 7         | 1.000    | 1.000    |
| 9         | 0.967    | 0.967    |

📊 Mejor resultado: **K=7 con Accuracy = 1.0**.

![Heatmap KNN con PCA](outputs/heatmap_knn_pca.png)

---

## 📌 Conclusiones

1. **Reducción eficiente**: Con solo 2 componentes se conserva >95% de la varianza.
2. **Visualización clara**: PCA facilita la exploración y comprensión de los datos en espacios reducidos.
3. **Desempeño óptimo con PCA+KNN**: La combinación alcanza un 100% de accuracy con K=7.
4. **Balance entre simplicidad y rendimiento**: PCA evita trabajar con todas las dimensiones sin sacrificar la calidad del modelo.

---
