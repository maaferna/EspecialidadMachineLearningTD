# setup_autoencoders.sh
set -e

# ── Estructura de carpetas ─────────────────────────────────────────────
mkdir -p scripts \
         src/utils \
         src/models \
         src/trainer \
         src/visualizer \
         outputs

# ── environment.yml (mínimo y compatible con CPU/GPU segun tu pip install) ──
cat > environment.yml << 'YAML'
name: especialidadmachinelearning
channels:
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

  # Deep Learning frameworks
  - tensorflow  # TensorFlow CPU/GPU (ver nota abajo)
  - keras
  - pytorch
  - torchvision
  - torchaudio  # útil si quieres audio en el futuro
  - cudatoolkit=12.1  # requerido para GPUs RTX 4090 (CUDA 12.x)

  # Experimentation / Hyperparameter optimization
  - optuna
  - scikit-optimize
  - hyperopt
  - deap
  - ray-tune

  # Utils & Dev
  - nbformat
  - pytest
  - kagglehub

  # pip extras
  - pip:
      - shap
      - lime
      - tensorflow-addons
      - tensorflow-datasets

YAML


echo "Proyecto listo. Activa el entorno con conda, instala deps y ejecuta:"
echo "  conda env update -f environment.yml"
echo "  conda activate especialidadmachinelearning"
echo "  python scripts/ae_main.py --mode both --epochs 20 --batch-size 128"
