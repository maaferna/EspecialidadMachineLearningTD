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
# 🧠 Proyecto: Comparación de Técnicas Avanzadas para Predicción de Ingresos

Este proyecto aplica y compara modelos avanzados de **regresión y clasificación** sobre el dataset **Adult Income**, con foco en precisión, estabilidad e interpretabilidad.

## 📁 Estructura del Proyecto

\`\`\`
.
├── data/                # Dataset 'adult' descargado desde OpenML
├── notebooks/           # Noteb#!/bin/bash

# Crear entorno Conda
echo "📦 Creando entorno conda 'especialidadmachinelearning'..."
conda env create -f environment.yml

# Activar entorno (solo aplica en terminal interactiva)
echo "⚠️  Para activar el entorno, ejecuta manualmente:"
echo "   conda activate especialidadmachinelearning"

# Crear carpetas necesarias
echo "📁 Creando estructura de carpetas..."
mkdir -p data outputs scripts src

# Crear módulos base
echo "📝 Generando módulos src/*.py..."
cat > src/preprocessing.py << EOF
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def cargar_y_preprocesar():
    from sklearn.datasets import fetch_openml
    df = fetch_openml("adult", version=2, as_frame=True).frame
    df = df.dropna()
    y = (df["class"] == ">50K").astype(int)
    X = df.drop(columns=["class"])
    X = pd.get_dummies(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
EOF

cat > src/modelos.py << EOF
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier
from sklearn.linear_model import QuantileRegressor

def modelo_random_forest():
    return RandomForestClassifier()

def modelo_xgboost():
    return XGBClassifier(use_label_encoder=False, eval_metric='logloss')

def modelo_elastic_net():
    return ElasticNet()

def modelo_cuantil(percentile):
    return QuantileRegressor(quantile=percentile / 100.0)
EOF

cat > src/evaluacion.py << EOF
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def evaluar_clasificacion(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }

def matriz_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
EOF

cat > src/visualizaciones.py << EOF
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def plot_matriz_confusion(conf_matrix, title="Confusion Matrix"):
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(f"outputs/{title.replace(' ', '_').lower()}.png")
    plt.close()

def plot_curva_roc(y_true, y_score, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f"outputs/{title.replace(' ', '_').lower()}.png")
    plt.close()
EOF

# Crear script main
echo "🚀 Generando scripts/main.py..."
cat > scripts/main.py << EOF
import sys
from pathlib import Path

# Ajuste para importar desde src/
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import cargar_y_preprocesar
from src.modelos import modelo_random_forest
from src.evaluacion import evaluar_clasificacion, matriz_confusion
from src.visualizaciones import plot_matriz_confusion, plot_curva_roc

from sklearn.model_selection import train_test_split

print("📊 Cargando y preprocesando datos...")
X, y = cargar_y_preprocesar()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("🧠 Entrenando modelo Random Forest...")
model = modelo_random_forest()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

resultados = evaluar_clasificacion(y_test, y_pred)
print("✅ Resultados:", resultados)

conf_matrix = matriz_confusion(y_test, y_pred)
plot_matriz_confusion(conf_matrix, title="Confusion Matrix Random Forest")
plot_curva_roc(y_test, y_prob, title="ROC Curve Random Forest")
EOF

echo "✅ Estructura lista. Ejecuta:"
echo "   conda activate especialidadmachinelearning"
echo "   python scripts/main.py"
ook generado al final
├── outputs/             # Imágenes, gráficas, métricas
├── scripts/
│   ├── main.py              # Script principal para entrenamiento y evaluación
│   └── crear_notebook.py    # Script para generar el notebook automáticamente
├── src/
│   ├── utils.py             # Funciones de carga y preprocesamiento
│   ├── modelos.py           # Implementación de ElasticNet, RF, XGBoost, etc.
│   ├── evaluador.py         # Métricas para regresión y clasificación
│   └── visualizador.py      # Visualización: curvas ROC, matrices, comparativas
├── environment.yml          # Entorno conda para reproducibilidad
└── README.md
\`\`\`

## ⚙️ Modelos Involucrados

- **Elastic Net**
- **Regresión Cuantílica** (percentiles 10, 50, 90)
- **Random Forest**
- **XGBoost**

## 🔍 Evaluación

- Clasificación: accuracy, matriz de confusión, curva ROC
- Regresión: RMSE, Pinball Loss

## 🚀 Instrucciones

1. Crear el entorno:

\`\`\`bash
conda env create -f environment.yml
conda activate especialidadmachinelearning
\`\`\`

2. Ejecutar el pipeline:

\`\`\`bash
python -m scripts.main
\`\`\`

3. Generar el notebook:

\`\`\`bash
python -m scripts.crear_notebook
\`\`\`

## 📌 Notas

- Dataset utilizado: \`fetch_openml("adult", version=2)\`
- Para regresión cuantílica: usar \`QuantileRegressor\` de \`scikit-learn >=1.1\`
- Pipeline modular, orientado a producción y análisis crítico

---

EOL

echo "✅ Proyecto creado exitosamente en: $(pwd)"
echo "➡️ Puedes comenzar con: conda activate especialidadmachinelearning"
