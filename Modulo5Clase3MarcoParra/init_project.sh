#!/bin/bash

# Nombre del proyecto
echo "📦 Iniciando proyecto: Comparación de Técnicas Avanzadas de Regresión..."


# Crear carpetas base
mkdir -p {data,notebooks,outputs,scripts,src}

# Crear archivos fuente base
touch src/{utils.py,modelos.py,evaluador.py,visualizador.py}
touch scripts/{main.py,crear_notebook.py}
touch README.md

# Crear environment.yml
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

# Iniciar README.md
cat <<EOF > /README.md
# 🔍 Boosting vs Bagging - Income Classification

Este proyecto compara técnicas de ensamblado (Boosting y Bagging) para predecir si una persona gana más de \$50K anuales utilizando el dataset **Adult Income**.

## 📁 Estructura

- \`scripts/\`: Contiene el \`main.py\` y notebook generator.
- \`src/\`: Módulos Python (preprocesamiento, modelos, evaluación, visualización).
- \`outputs/\`: Resultados como CSV, imágenes y métricas.
- \`models/\`: Modelos serializados.
- \`visuals/\`: Gráficos generados.
- \`environment.yml\`: Entorno Conda reproducible.

## ⚙️ Setup

```bash
conda env create -f environment.yml
conda activate $ENV_NAME
python scripts/main.py
