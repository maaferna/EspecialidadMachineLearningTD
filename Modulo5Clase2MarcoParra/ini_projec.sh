#!/bin/bash

echo "📦 Iniciando proyecto: Comparación de Técnicas Avanzadas de Regresión..."

# Crear carpetas base
mkdir -p {data,notebooks,outputs,scripts,src}

# Crear archivos fuente base
touch src/{utils.py,modelos.py,evaluador.py,visualizador.py}
touch scripts/{main.py,crear_notebook.py}
touch README.md

# Crear environment.yml
if [ ! -f environment.yml ]; then
  echo "🧪 Generando archivo environment.yml..."
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
  echo "📎 Ya existe un environment.yml, no se sobrescribe."
fi

# README inicial
cat <<EOL > README.md
# Proyecto: Comparación de Técnicas Avanzadas de Regresión

Este proyecto implementa tres enfoques distintos:

1. **Elastic Net** para predicción de precios de viviendas.
2. **Regresión Cuantílica** para estimar percentiles de ingresos.
3. **VAR** para proyección de indicadores macroeconómicos.

La estructura está modularizada con scripts de carga, modelamiento y visualización, facilitando su mantenimiento y escalabilidad.

## Estructura del Proyecto

- \`src/\`: Módulos reutilizables.
- \`scripts/\`: Ejecución principal y generación de notebooks.
- \`outputs/\`: Resultados, métricas y visualizaciones.
- \`notebooks/\`: Versión interactiva del flujo.

## Activación del entorno

\`\`\`bash
conda env create -f environment.yml
conda activate regresion_avanzada
\`\`\`

## Ejecución principal

\`\`\`bash
python -m scripts.main
\`\`\`

EOL

echo "✅ Estructura y entorno inicial listos."
echo "➡️ Ejecuta: conda activate regresion_avanzada"