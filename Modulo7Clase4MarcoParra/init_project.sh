# bootstrap_tl_project.sh
#!/usr/bin/env bash
set -e

# 1) Carpetas
mkdir -p scripts
mkdir -p src/{data,models,trainer,visualizer,utils}
mkdir -p data
mkdir -p outputs_tl

# 2) environment.yml (opcional; ajusta TF según tu setup)
cat > environment.yml << 'YAML'
name: especialidadmachinelearning
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  # Core
  - python=3.10
  - pip
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost
  - lightgbm
  - catboost
  - hdbscan
  - umap-learn


  # HPO / Experimentation
  - optuna
  - scikit-optimize
  - hyperopt
  - deap
  - ray-tune

  # Utils & Dev
  - nbformat
  - pytest
  - kagglehub

  # Cosas que van mejor por pip (TF GPU y extras)
  - pip:
      - tensorflow[and-cuda]==2.15.*
      - shap
      - lime

YAML

echo "✓ Estructura creada. Opcional: conda env create -f environment.yml"
