#!/bin/bash

# Crear estructura de carpetas en el directorio actual
mkdir -p src scripts outputs notebook

# Crear archivos vacíos base
touch src/{funciones.py,visualizador.py,clasificador.py}
touch scripts/{main.py,crear_notebook.py}
touch README.md environment.yml

# Agregar contenido base al README
cat <<EOL > README.md
# Optimización y Análisis Geométrico de Funciones en Dos Variables

Este proyecto implementa el análisis simbólico, visual y numérico de una función g(x, y) con dos variables.
Se calcula el gradiente, la matriz Hessiana y se visualizan los puntos críticos. También se discute la relación
con técnicas de optimización utilizadas en Machine Learning.
EOL

# Crear environment.yml con las dependencias requeridas
cat <<EOL > environment.yml
name: especialidadmachinelearning
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - numpy=1.23.5
  - numba=0.56.4
  - matplotlib
  - nbformat
  - sympy
  - scipy
  - pip
  - pip:
      - tqdm
      - scikit-learn
      - pandas
      - seaborn
EOL

echo "✅ Estructura y archivos creados en $(pwd)"
