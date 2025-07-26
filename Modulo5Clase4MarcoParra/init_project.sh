#!/bin/bash

echo "💼 Iniciando estructura para: Predicción de Ingresos (Adult Dataset)..."

# Crear carpetas del proyecto en el directorio actual
mkdir -p {data,notebooks,outputs,scripts,src}

# Archivos fuente
touch src/{utils.py,modelos.py,evaluador.py,visualizador.py}
touch scripts/{main.py,crear_notebook.py}
touch README.md

# Crear environment.yml si no existe
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
EOF
  echo "✅ Archivo environment.yml creado."
else
  echo "📦 environment.yml ya existe, se utilizará el entorno existente."
fi

# README con estructura e instrucciones
cat <<EOL > README.md
# Proyecto de Predicción de Ingresos (Adult Dataset)

EOL

echo "✅ Proyecto creado exitosamente en: $(pwd)"
echo "➡️ Puedes comenzar con: conda activate especialidadmachinelearning"
