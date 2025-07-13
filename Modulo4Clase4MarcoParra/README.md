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

Optimizar los hiperparámetros de `RandomForestClassifier` usando un **algoritmo genético** con la librería deap y comparar los resultados con modelos de optimizaciòn tradicionales.


Optimizar modelos predictivos de enfermedades crónicas mediante diferentes estrategias de búsqueda de hiperparámetros, con foco en el rendimiento evaluado por métricas como AUC, F1, Precisión y Recall.

---

## 📈 Comparación de Métricas por Modelo

![Comparación de métricas](outputs/comparacion_metricas_modelos.png)

---

## 📊 Resultados por Método

| Modelo         | ROC Curve                      | Matriz de Confusión               |
|----------------|--------------------------------|-----------------------------------|
| Base           | ![](outputs/roc_base.png)      | ![](outputs/matriz_confusion_base.png) |
| Genético       | ![](outputs/roc_genético.png)  | ![](outputs/matriz_confusion_genético.png) |
| GridSearch     | ![](outputs/roc_gridsearch.png)| ![](outputs/matriz_confusion_gridsearch.png) |
| RandomSearch   | ![](outputs/roc_randomsearch.png)| ![](outputs/matriz_confusion_randomsearch.png) |
| Optuna         | ![](outputs/roc_optuna.png)    | ![](outputs/matriz_confusion_optuna.png) |
| Skopt          | ![](outputs/roc_skopt.png)     | ![](outputs/matriz_confusion_skopt.png) |

---

## 🔍 Análisis de Resultados

- **Precisión general**: todos los modelos optimizados mantuvieron un AUC de 0.99, lo cual indica que la capacidad discriminativa se mantuvo alta en todos los casos.

- **F1-Score y Recall**: el modelo optimizado con **Optuna** fue el que logró el mejor balance entre precisión (0.95) y recall (0.97), alcanzando un F1-score de 0.96, ideal para contextos donde los falsos negativos tienen alto costo.

- **Modelos Genético, Grid y RandomSearch** ofrecieron resultados estables y prácticamente equivalentes en métricas principales, lo que refleja que incluso métodos clásicos siguen siendo competitivos.

- **Scikit-Optimize (Skopt)** resultó muy eficiente, manteniendo todas las métricas en 0.94–0.99, y representa una alternativa liviana para exploración rápida.

- **Errores de clasificación**: observando las matrices de confusión, la cantidad de falsos positivos y falsos negativos se mantuvo baja y equilibrada en todos los métodos. Optuna logró reducir los falsos negativos a solo 3 casos.

El uso de técnicas de optimización permite refinar significativamente modelos base en problemas de clasificación multiclase complejos. Herramientas como Optuna y Ray Tune permiten explorar espacios de búsqueda de manera más eficiente que los métodos tradicionales, reduciendo tiempos y mejorando resultados. En contextos sensibles como la salud pública, maximizar el recall es crucial, y este análisis muestra cómo la optimización puede lograrlo sin comprometer la precisión.

---

