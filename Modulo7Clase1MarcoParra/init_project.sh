#!/bin/bash

# Crear carpetas base directamente en el directorio actual
mkdir -p {data,notebooks,outputs,tests}
mkdir -p scripts
mkdir -p src/{utils,models,visualizer,evaluator}

# Crear archivos iniciales
touch scripts/main.py
touch src/utils/{__init__.py,data_loader.py,preprocessing.py}
touch src/models/{__init__.py,base_model.py,nn_model.py}
touch src/visualizer/{__init__.py,plots.py}
touch src/evaluator/{__init__.py,metrics.py}

# Crear README
cat <<EOT >> README.md
# Especialidad Machine Learning Project

Estructura inicial generada automáticamente.
- **data/** → datasets crudos o procesados.
- **notebooks/** → experimentos en Jupyter.
- **outputs/** → modelos entrenados, gráficos, reportes.
- **scripts/** → \`main.py\` punto de entrada.
- **src/** → código modular.
  - utils → carga y preprocesamiento de datos.
  - models → definición de modelos.
  - visualizer → funciones de gráficos.
  - evaluator → evaluación de modelos.
EOT

# Crear environment.yml para conda
cat <<EOT >> environment.yml
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
      - kagglehub
      - umap-learn
      - tensorflow
      - keras
      - pytorch
      - torchvision
EOT

echo "✅ Proyecto creado en $(pwd)"
