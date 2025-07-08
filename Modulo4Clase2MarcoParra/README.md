# 🧪 Proyecto: DiabetesHiperparametros

Este proyecto forma parte de la especialidad en Machine Learning.  
Utiliza el entorno compartido **`especialidadmachinelearning`** basado en Conda.

## 🎯 Objetivo

Clasificar la presencia de diabetes usando Random Forest,  
optimizando hiperparámetros con **Grid Search** y **Random Search**,  
y comparando métricas como F1-Score, AUC, precisión y tiempo de entrenamiento.

## 🚀 Instrucciones de uso

### 1. Activar entorno Conda
```bash
conda activate especialidadmachinelearning
```

### 2. Ejecutar pipeline principal
```bash
python -m scripts.main
```

### 3. Generar notebook automáticamente
```bash
python -m scripts.crear_notebook
```

## 📁 Estructura del proyecto

```
├── data/                # Dataset CSV original
├── notebooks/           # Notebooks generados automáticamente
├── outputs/             # Gráficos y métricas generadas
├── scripts/
│   ├── main.py
│   └── crear_notebook.py
├── src/
│   ├── utils.py
│   ├── modelos.py
│   ├── optimizacion.py
│   └── visualizador.py
├── environment.yml
└── README.md
```

## 📦 Librerías clave

- pandas, numpy, matplotlib, seaborn
- scikit-learn
- optuna, scikit-optimize, hyperopt
- nbformat, pytest

> ⚠️ Asegúrate de tener activo el entorno `especialidadmachinelearning` antes de ejecutar los scripts.

