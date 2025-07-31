#!/bin/bash

echo "🏦 Creando estructura del proyecto: Scoring Crediticio con Interpretabilidad..."

# Nombre del entorno
ENV_NAME="especialidadmachinelearning"

# Crear carpetas
mkdir -p {data,notebooks,outputs,scripts,src,docs}

# Archivos principales
touch scripts/{main.py,crear_notebook.py}
touch src/{utils.py,modelos.py,evaluador.py,visualizador.py,interpretabilidad.py}
touch README.md
touch environment.yml

# Archivo environment.yml para conda
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
  - xgboost
  - pip
  - pip:
      - optuna
      - scikit-optimize
      - hyperopt
      - deap
      - nbformat
      - pytest
      - ray[tune]
      - lightgbm
      - catboost

EOF

# README inicial
cat <<EOL > README.md
# 📌 Proyecto: Scoring Crediticio con Interpretabilidad

## 🎯 Objetivo
Construir un modelo predictivo para el scoring crediticio usando técnicas de regularización
(Lasso/Ridge) y aplicar interpretabilidad con **SHAP** y **LIME**.

## 📂 Estructura del Proyecto
\`\`\`
├── data/                # Dataset raw y procesados
├── notebooks/           # Jupyter Notebooks
├── outputs/             # Gráficos, reportes y resultados
├── scripts/
│   ├── main.py          # Pipeline principal
│   └── crear_notebook.py # Generar notebook desde main
├── src/
│   ├── utils.py         # Carga y preprocesamiento
│   ├── modelos.py       # Definición y entrenamiento de modelos
│   ├── evaluador.py     # Evaluación de métricas
│   ├── visualizador.py  # Visualizaciones
│   └── interpretabilidad.py # SHAP y LIME
├── docs/                # Informe técnico
├── environment.yml      # Dependencias conda
└── README.md
\`\`\`

## ⚙️ Flujo del Proyecto
1. **Carga y preprocesamiento** del dataset \`credit\` de OpenML.
2. **Entrenamiento** con regresión logística y Random Forest, aplicando Lasso y Ridge.
3. **Evaluación** con Accuracy, Recall, F1 y AUC.
4. **Interpretabilidad** usando SHAP y LIME.
5. **Visualización** de métricas y explicaciones.
6. **Análisis crítico** y elaboración del informe técnico.

## 🚀 Cómo iniciar
\`\`\`bash
bash setup_scoring_project.sh
conda env create -f environment.yml
conda activate $ENV_NAME
python -m scripts.main
\`\`\`

EOL

echo "✅ Proyecto creado exitosamente en: $(pwd)"
echo "➡️ Ahora puedes activar el entorno con: conda env create -f environment.yml && conda activate $ENV_NAME"
