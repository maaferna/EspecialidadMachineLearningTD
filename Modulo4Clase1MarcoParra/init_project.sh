#!/bin/bash

echo "🚀 Inicializando estructura del proyecto Predicción de Diabetes..."

# Crear carpetas base
mkdir -p src scripts outputs notebook data

# Descargar dataset Pima Indians
wget -O data/pima-indians-diabetes.csv https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

# Crear environment.yml
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
      - nbformat
      - pytest
EOF

echo "✅ Archivo environment.yml generado."

# Crear archivos iniciales
touch src/utils.py src/modelos.py src/optimizacion.py src/visualizador.py
touch scripts/main.py scripts/crear_notebook.py

# Hacer scripts ejecutables
chmod +x scripts/*.py
chmod +x init_project.sh

echo "📦 Entorno conda 'especialidadmachinelearning' será creado con:"
echo "👉 conda env create -f environment.yml"

echo "✅ Proyecto inicializado correctamente."
echo "🔁 Recuerda activar el entorno con: conda activate especialidadmachinelearning"
