#!/bin/bash

echo "🧬 Iniciando estructura para: AjusteGeneticoRF..."

# Crear carpetas del proyecto en el directorio actual
mkdir -p {data,notebooks,outputs,scripts,src}

# Archivos fuente
touch src/{utils.py,modelos.py,visualizador.py,optimizador.py,evaluador.py}
touch scripts/{main.py,crear_notebook.py}
touch README.md

# Crear environment.yml solo si no existe
if [ ! -f environment.yml ]; then
  echo "🧪 Generando environment.yml base para entorno conda..."
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
  - pip
  - pip:
      - scikit-optimize
      - hyperopt
      - optuna
      - deap
      - nbformat
      - pytest
EOF
  echo "✅ Archivo environment.yml creado."
else
  echo "📦 environment.yml ya existe, se utilizará el entorno compartido."
fi

# README con instrucciones iniciales
cat <<EOL > README.md
# 🧬 Proyecto: Ajuste de Hiperparámetros con Algoritmos Genéticos

Este proyecto aplica un algoritmo genético para optimizar los hiperparámetros de un modelo RandomForest, utilizando el dataset de cáncer de mama incluido en `scikit-learn`.

## 📁 Estructura del Proyecto

\`\`\`
.
├── data/                # Archivos persistentes si fuera necesario
├── notebooks/           # Jupyter notebooks generados
├── outputs/             # Gráficos y métricas generadas
├── scripts/
│   ├── main.py          # Pipeline de entrenamiento
│   └── crear_notebook.py# Generador automático de notebook
├── src/
│   ├── genetico.py      # Lógica del algoritmo genético con DEAP
│   ├── modelos.py       # Modelo base y optimizado
│   ├── utils.py         # Preprocesamiento y carga
│   └── visualizador.py  # Visualización de métricas
├── environment.yml      # Entorno compartido con la especialidad
└── README.md
\`\`\`

## 🚀 Instrucciones para Ejecutar

1. Activar entorno:
\`\`\`bash
conda activate especialidadmachinelearning
\`\`\`

2. Ejecutar el pipeline:
\`\`\`bash
python -m scripts.main
\`\`\`

3. Generar el notebook:
\`\`\`bash
python -m scripts.crear_notebook
\`\`\`

## 🧠 Objetivo

Optimizar los hiperparámetros de \`RandomForestClassifier\` usando un **algoritmo genético** con la librería `DEAP`.

## 🔬 Dataset

- `load_breast_cancer()` desde `sklearn.datasets`

## 📈 Métricas de Evaluación

- `F1-score` con `cross_val_score` como función de aptitud
- `classification_report` para evaluación final

## 📌 Reflexión Final

Se incluirá un análisis comparativo entre el modelo base y el optimizado con algoritmos genéticos.

---

EOL

echo "✅ Proyecto creado exitosamente en: $(pwd)"
echo "➡️ Puedes comenzar con: conda activate especialidadmachinelearning"
