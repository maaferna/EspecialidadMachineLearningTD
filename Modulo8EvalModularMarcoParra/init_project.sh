#!/usr/bin/env bash
set -euo pipefail

# Uso:
#   chmod +x bootstrap_estructura_clasificacion.sh
#   ./bootstrap_estructura_clasificacion.sh
# Crea la estructura en el directorio ACTUAL (sin subcarpeta).

mk()  { mkdir -p "$1"; }
mkf() { mkdir -p "$(dirname "$1")"; : > "$1"; }

echo "🧱 Creando estructura del proyecto de CLASIFICACIÓN (NLP clínico, ética y sesgos)..."

# === DATA LIFECYCLE ===
mk data/raw data/interim data/processed data/external
mk models/checkpoints models/artifacts
mk reports/figures reports/tables reports/pdf reports/fairness reports/explainability reports/logs
mk notebooks
mk configs
mk docs docs/ethics

# === SRC (módulos Python) ===
mk src/{data,preprocessing,features,models,evaluation,explainability,visualization,utils,pipelines}

# Paquetes Python (__init__.py)
mkf src/__init__.py
for pkg in data preprocessing features models evaluation explainability visualization utils pipelines; do
  mkf "src/${pkg}/__init__.py"
done

# Placeholders de módulos (VACÍOS)
# Data
mkf src/data/datasets.py
mkf src/data/splits.py

# Preprocessing
mkf src/preprocessing/cleaning.py
mkf src/preprocessing/tokenize_spacy.py
mkf src/preprocessing/tokenize_nltk.py
mkf src/preprocessing/stopwords.py

# Features
mkf src/features/vectorize_tfidf.py
mkf src/features/embeddings.py

# Models (clasificación + mitigación de sesgos)
mkf src/models/train_classifier.py
mkf src/models/predict.py
mkf src/models/bias_mitigation.py

# Evaluation (métricas + fairness + calibración)
mkf src/evaluation/metrics.py
mkf src/evaluation/fairness.py
mkf src/evaluation/calibration.py

# Explainability
mkf src/explainability/shap_utils.py
mkf src/explainability/lime_utils.py

# Visualization
mkf src/visualization/plots.py
mkf src/visualization/confusion_matrix.py

# Pipelines
mkf src/pipelines/run_train.py
mkf src/pipelines/run_infer.py
mkf src/pipelines/run_fairness_audit.py
mkf src/pipelines/run_explainability.py

# Utils
mkf src/utils/io.py
mkf src/utils/config.py
mkf src/utils/logging.py
mkf src/utils/seed.py

# === SCRIPTS de entrada (VACÍOS) ===
mk scripts
mkf scripts/main_train.py
mkf scripts/main_infer.py
mkf scripts/evaluate_bias.py
mkf scripts/download_small_clinical_notes.py
mkf scripts/make_report_pdf.py

# === CONFIGS (VACÍOS) ===
mkf configs/config_default.yaml
mkf configs/train_config.yaml
mkf configs/fairness_config.yaml
mkf configs/label_mapping.yaml

# === NOTEBOOKS (VACÍOS) ===
mkf notebooks/00_setup.ipynb
mkf notebooks/01_train_eval.ipynb
mkf notebooks/02_fairness_bias.ipynb
mkf notebooks/03_explainability.ipynb

# === REPORTING / DOCUMENTACIÓN (VACÍOS) ===
mkf reports/fairness/README.md
mkf reports/explainability/README.md
mkf reports/logs/.gitkeep

# Ética / Gobernanza (plantillas vacías)
mkf docs/ethics/model_card.md
mkf docs/ethics/datasheet_for_dataset.md
mkf docs/ethics/bias_mitigation_plan.md
mkf docs/ethics/fairness_checklist.md
mkf docs/ethics/privacy_and_security.md
mkf docs/ethics/limitations_and_risks.md
mkf docs/ethics/monitoring_and_feedback.md

# === ENV / METADATOS (VACÍOS) ===
mk env tests
mkf env/environment.yml
mkf .gitignore
mkf README.md
mkf LICENSE
mkf pyproject.toml
mkf setup.cfg

echo "✅ Estructura creada en: $(pwd)"
echo "Siguiente paso: completa env/environment.yml (Conda) y luego rellenamos los módulos."
