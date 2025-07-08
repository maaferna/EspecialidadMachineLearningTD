#!/bin/bash

echo "🚀 Inicializando estructura del proyecto..."

# Crear estructura de carpetas
mkdir -p src scripts outputs notebook

# Generar environment.yml automáticamente
cat <<EOF > environment.yml
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
      - pytest
EOF

echo "✅ Archivo environment.yml generado."

# Crear archivos base
touch src/utils.py src/optimizadores.py src/visualizador.py
touch scripts/main.py scripts/crear_notebook.py

# Hacer scripts ejecutables
chmod +x scripts/*.py
chmod +x init_project.sh

# Crear entorno Conda
echo "📦 Creando entorno conda 'especialidadmachinelearning'..."
conda env create -f environment.yml

echo "✅ Proyecto inicializado correctamente."
echo "👉 Activa el entorno con: conda activate especialidadmachinelearning"
