#!/bin/bash

echo "📦 Creando proyecto: Modulo 5 Evaluacion Modular..."

# Crear carpetas principales
mkdir -p {data,notebooks,outputs,src,scripts}

# Archivos fuente
touch src/{utils.py,modelos.py,visualizador.py,evaluador.py}
touch scripts/{main.py,crear_notebook.py}
touch README.md

# Crear environment.yml
cat <<EOL > environment.yml
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
      - hdbscan
      - umap-learn
EOL

echo "✅ Proyecto creado exitosamente."
