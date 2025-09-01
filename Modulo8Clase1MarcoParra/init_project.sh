#!/usr/bin/env bash
# bootstrap_nlp_clinical_similarity.sh
# Crea la estructura del proyecto directamente en el directorio actual.

set -e

echo "📁 Creando estructura de proyecto en $(pwd)..."

# ─────────────────────────── carpetas base ───────────────────────────
mkdir -p envs
mkdir -p data/{raw,processed,external}
mkdir -p notebooks
mkdir -p scripts
mkdir -p outputs/{figures,reports}
mkdir -p tests
mkdir -p src/{data,preprocess,features,similarity,visualizer,utils}

# ─────────────────────────── archivos raíz ───────────────────────────
touch README.md
touch .gitignore

# environment.yml con contenido
cat > envs/environment.yml << 'EOF'
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
EOF

# ─────────────────────────── notebooks ───────────────────────────────
touch notebooks/clinical_similarity_colab.ipynb

# ─────────────────────────── data placeholders ───────────────────────
touch data/raw/notes_sample.txt
touch data/README.md

# ─────────────────────────── scripts (CLI / orquestación) ────────────
touch scripts/main.py
touch scripts/prepare_corpus.py
touch scripts/run_all.sh

# ─────────────────────────── src (paquetes Python) ───────────────────
touch src/__init__.py
touch src/data/__init__.py
touch src/preprocess/__init__.py
touch src/features/__init__.py
touch src/similarity/__init__.py
touch src/visualizer/__init__.py
touch src/utils/__init__.py

# Archivos vacíos para implementación futura
touch src/data/loader.py
touch src/preprocess/clean.py
touch src/features/vectorize.py
touch src/similarity/metrics.py
touch src/visualizer/plots.py
touch src/utils/io.py

# ─────────────────────────── tests ───────────────────────────────────
touch tests/test_smoke.py

echo "✔ Proyecto listo."
echo "Siguiente paso:"
echo "  conda env create -f envs/environment.yml"
echo "  conda activate especialidadmachinelearning"
