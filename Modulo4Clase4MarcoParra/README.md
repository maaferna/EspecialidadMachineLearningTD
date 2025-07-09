# 🧬 Proyecto: Ajuste de Hiperparámetros con Algoritmos Genéticos

Este proyecto aplica un algoritmo genético para optimizar los hiperparámetros de un modelo RandomForest, utilizando el dataset de cáncer de mama incluido en .

## 📁 Estructura del Proyecto

```
.
├── data/                # Archivos persistentes si fuera necesario
├── notebooks/           # Jupyter notebooks generados
├── outputs/             # Gráficos y métricas generadas
├── scripts/
│   ├── main.py          # Pipeline de entrenamiento
│   └── crear_notebook.py# Generador automático de notebook
├── src/
├   ├─ optimizador.py      # Optimizadores
│   ├── optimizador.py      # Lógica de optimizacion
│   ├── modelos.py       # Modelo base y optimizado
│   ├── utils.py         # Preprocesamiento y carga
│   └── visualizador.py  # Visualización de métricas
├── environment.yml      # Entorno compartido con la especialidad
└── README.md
```

## 🚀 Instrucciones para Ejecutar

1. Activar entorno:
```bash
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

## 🧠 Objetivo

Optimizar los hiperparámetros de `RandomForestClassifier` usando un **algoritmo genético** con la librería .

## 🔬 Dataset

-  desde 

## 📈 Métricas de Evaluación

-  con  como función de aptitud
-  para evaluación final

## 📌 Reflexión Final

Se incluirá un análisis comparativo entre el modelo base y el optimizado con algoritmos genéticos.

---

