#!/bin/bash

echo "🚀 Inicializando estructura del proyecto de Optimización Bayesiana en Salud..."

# Crear estructura de carpetas
mkdir -p src scripts notebook outputs tests

# Generar environment.yml automáticamente
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

# Crear archivos base
touch src/utils.py
touch src/visualizador.py
touch scripts/main.py
touch scripts/crear_notebook.py

# Agregar init para convertir src en paquete
touch src/__init__.py

# Hacer scripts ejecutables
chmod +x scripts/*.py
chmod +x init_project.sh

echo "✅ Estructura creada correctamente."
echo "📄 Para crear el entorno Conda, ejecuta:"
echo "   conda env create -f environment.yml"
echo "👉 Luego activa con:"
echo "   conda activate optimizacion_bayesiana_salud"
