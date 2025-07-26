#!/bin/bash

echo "📁 Creando estructura de proyecto para Regresión con Regularización..."

mkdir -p {data,notebooks,outputs,scripts,src}
touch scripts/{main.py,crear_notebook.py}
touch src/{utils.py,modelos.py,evaluador.py,visualizador.py}
touch README.md

# Environment
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
  - jupyter
  - pip
  - pip:
      - jupytext
      - nbformat
EOF

echo "✅ Estructura creada y environment.yml generado."

