#!/bin/bash

# Crear estructura de carpetas en el directorio actual
mkdir -p src scripts notebooks outputs

# Crear archivos base
touch src/utils.py
touch src/modelos.py
touch src/evaluador.py
touch src/visualizador.py
touch scripts/main.py
touch notebooks/README.md
touch environment.yml

# Crear environment.yml para Conda
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
EOL

echo "✅ Proyecto inicializado con éxito."
echo "Usa: conda env create -f environment.yml"
