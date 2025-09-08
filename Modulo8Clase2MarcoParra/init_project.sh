#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   chmod +x bootstrap_estructura.sh
#   ./bootstrap_estructura.sh
# Crea estructura en el *directorio actual* (no subcarpeta).

mk()  { mkdir -p "$1"; }
mkf() { mkdir -p "$(dirname "$1")"; : > "$1"; }

# Carpetas principales
mk data/raw data/interim data/processed data/external
mk notebooks
mk reports/figures
mk configs
mk docs
mk tests
mk env
mk scripts
mk src/{data,preprocessing,features,models,evaluation,visualization,utils,pipelines}

# __init__.py para paquetes Python (vacíos)
mkf src/__init__.py
for pkg in data preprocessing features models evaluation visualization utils pipelines; do
  mkf "src/${pkg}/__init__.py"
done

# Placeholders de módulos (vacíos)
mkf src/data/datasets.py
mkf src/preprocessing/cleaning.py
mkf src/preprocessing/tokenize_spacy.py
mkf src/preprocessing/tokenize_nltk.py
mkf src/preprocessing/stopwords.py
mkf src/features/vectorize_tfidf.py
mkf src/evaluation/corpus_stats.py
mkf src/visualization/term_bars.py
mkf src/pipelines/run_pipeline.py
mkf src/utils/io.py

# Scripts y notebooks (vacíos)
mkf scripts/main.py
mkf scripts/download_small_clinical_notes.py
mkf notebooks/01_pipeline_demo.ipynb

# Config y metadatos (vacíos)
mkf configs/config_default.yaml
mkf env/environment.yml
mkf .gitignore
mkf README.md
mkf LICENSE
mkf pyproject.toml
mkf setup.cfg

# Alias opcional por compatibilidad
mk srs
mkf srs/README.md

echo "✅ Estructura creada en: $(pwd)"
echo "👉 Ahora pega el environment.yml provisto en env/environment.yml"
