#!/bin/bash

# Crear carpetas en el root
mkdir -p scripts
mkdir -p src
mkdir -p outputs
mkdir -p notebooks

# Crear archivos principales en el root
touch scripts/main.py
touch src/utils.py
touch src/modelos.py
touch src/visualizador.py
touch src/evaluador.py
touch notebooks/analisis.ipynb
touch README.md

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
EOL

# Mensaje de confirmación
echo "✅ Estructura del proyecto creada en el root actual."
echo "Carpetas: scripts/, src/, outputs/, notebooks/"
echo "Archivos iniciales listos: main.py, utils.py, modelos.py, visualizador.py, evaluador.py, environment.yml, README.md, analisis.ipynb"
echo "Para crear el entorno Conda ejecuta:"
echo "conda env create -f environment.yml"
