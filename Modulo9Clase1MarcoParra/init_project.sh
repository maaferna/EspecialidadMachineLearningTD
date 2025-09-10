#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${PWD}"
echo ">> Inicializando proyecto en: ${ROOT_DIR}"

# -------------------------------
# Carpetas
# -------------------------------
mkdir -p configs \
         data/raw data/processed \
         models \
         reports/figures/explainability reports/logs reports/runs \
         notebooks \
         src/data src/features src/models src/evaluation src/explain src/utils \
         scripts

# -------------------------------
# .gitignore
# -------------------------------
cat > .gitignore <<'EOF'
__pycache__/
.ipynb_checkpoints/
.env
.venv
*.pyc
*.pyo
*.pyd
*.DS_Store
models/
reports/runs/
reports/figures/explainability/*.html
EOF

# -------------------------------
# requirements.txt (opcional; puedes usar tu environment.yml actual)
# -------------------------------
cat > requirements.txt <<'EOF'
pandas
numpy
scikit-learn
matplotlib
seaborn
lime
shap
datasets
EOF

# -------------------------------
# Config por defecto
# -------------------------------
cat > configs/config_default.yaml <<'EOF'
project:
  seed: 42
  output_dir: reports
  logs_dir: reports/logs
  figures_dir: reports/figures
  explain_dir: reports/figures/explainability
  runs_dir: reports/runs

dataset:
  # options: huggingface | csv | synthetic
  source: huggingface
  # 🤗 datasets: imdb (25k train / 25k test). Para demo tomamos un subset.
  hf_name: imdb
  text_col: text
  label_col: label    # 0=negativo, 1=positivo
  max_samples_train: 2000   # reduce para demo rápida
  max_samples_test:  1000
  csv_path: data/raw/opiniones.csv  # si source=csv
  simulate_min: 200                 # si source=synthetic

preprocessing:
  lowercase: true
  remove_punct: true
  remove_numbers: false
  stopwords: english               # simple: 'english' de sklearn
  ngram_range: [1,2]
  max_features: 20000

model:
  # options: logreg | random_forest
  name: logreg
  logreg:
    C: 1.0
    class_weight: balanced
    max_iter: 200
  random_forest:
    n_estimators: 300
    max_depth: null
    class_weight: balanced

training:
  test_size: 0.2
  stratify: true
  scoring: f1

explain:
  lime_samples: 2
  shap_samples: 2
  topk: 10
EOF

# -------------------------------
# src/utils/io.py
# -------------------------------
cat > src/utils/io.py <<'EOF'
from __future__ import annotations
from pathlib import Path
import json

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_yaml(path: str | Path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
EOF

# -------------------------------
# src/data/datasets.py
# -------------------------------
cat > src/data/datasets.py <<'EOF'
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple

def load_dataset(cfg) -> Tuple[pd.DataFrame, str, str]:
    src = cfg["dataset"]["source"]
    text_col = cfg["dataset"]["text_col"]
    label_col = cfg["dataset"]["label_col"]

    if src == "huggingface":
        from datasets import load_dataset
        ds = load_dataset(cfg["dataset"]["hf_name"])
        train = ds["train"].to_pandas()
        test  = ds["test"].to_pandas()
        # opcional: subset para rapidez
        mtr = cfg["dataset"].get("max_samples_train")
        mts = cfg["dataset"].get("max_samples_test")
        if mtr: train = train.sample(n=min(mtr, len(train)), random_state=42)
        if mts: test  = test.sample(n=min(mts, len(test)),  random_state=42)
        df = pd.concat([train, test], ignore_index=True)
        df = df[[text_col, label_col]].dropna()
        return df, text_col, label_col

    elif src == "csv":
        path = Path(cfg["dataset"]["csv_path"])
        if not path.exists():
            raise FileNotFoundError(f"No existe CSV en {path}")
        df = pd.read_csv(path)
        if not {text_col, label_col}.issubset(df.columns):
            raise ValueError(f"CSV debe tener columnas {text_col}, {label_col}")
        return df, text_col, label_col

    elif src == "synthetic":
        n = int(cfg["dataset"].get("simulate_min", 200))
        np.random.seed(42)
        pos = [
            "Excelente atención, personal amable.",
            "Muy buena experiencia, tratamiento efectivo.",
            "Me sentí acompañado y escuchado.",
            "Rápida atención y soluciones claras."
        ]
        neg = [
            "Mala atención, no volvería.",
            "Me ignoraron, pésima experiencia.",
            "Larga espera y respuestas confusas.",
            "No resolvieron mi problema."
        ]
        rows = []
        for i in range(n):
            if i % 2 == 0:
                rows.append({"text": np.random.choice(pos), "label": 1})
            else:
                rows.append({"text": np.random.choice(neg), "label": 0})
        df = pd.DataFrame(rows)
        return df, "text", "label"

    else:
        raise ValueError(f"dataset.source no soportado: {src}")
EOF

# -------------------------------
# src/features/text.py
# -------------------------------
cat > src/features/text.py <<'EOF'
from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def simple_clean(s: str, lowercase=True, remove_punct=True, remove_numbers=False):
    if lowercase:
        s = s.lower()
    if remove_punct:
        s = re.sub(r"[^\w\sáéíóúüñ]", " ", s, flags=re.UNICODE)
    if not remove_numbers:
        return re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\d+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def build_vectorizer(cfg) -> TfidfVectorizer:
    prep = cfg["preprocessing"]
    stop = prep.get("stopwords", "english")
    ngram = tuple(prep.get("ngram_range", [1,2]))
    maxf  = prep.get("max_features", 20000)

    def preproc(x): return simple_clean(
        x,
        lowercase=prep.get("lowercase", True),
        remove_punct=prep.get("remove_punct", True),
        remove_numbers=prep.get("remove_numbers", False),
    )

    return TfidfVectorizer(
        preprocessor=preproc,
        stop_words=stop,
        ngram_range=ngram,
        max_features=maxf
    )
EOF

# -------------------------------
# src/models/train.py
# -------------------------------
cat > src/models/train.py <<'EOF'
from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def split_xy(df, text_col, label_col, test_size=0.2, stratify=True, seed=42):
    y = df[label_col].astype(int).to_numpy()
    X = df[text_col].astype(str).tolist()
    return train_test_split(X, y, test_size=test_size, stratify=y if stratify else None, random_state=seed)

def get_model(cfg):
    name = cfg["model"]["name"].lower()
    if name == "logreg":
        p = cfg["model"]["logreg"]
        return LogisticRegression(
            C=float(p.get("C",1.0)),
            class_weight=p.get("class_weight","balanced"),
            max_iter=int(p.get("max_iter",200)),
            n_jobs=-1
        )
    elif name == "random_forest":
        p = cfg["model"]["random_forest"]
        return RandomForestClassifier(
            n_estimators=int(p.get("n_estimators",300)),
            max_depth=p.get("max_depth", None),
            class_weight=p.get("class_weight","balanced"),
            n_jobs=-1,
            random_state=42
        )
    else:
        raise ValueError(f"Modelo no soportado: {name}")

def evaluate(y_true, y_pred, labels=("neg","pos")):
    cm = confusion_matrix(y_true, y_pred).tolist()
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    return {"confusion": cm, "report": rep, "accuracy": acc, "f1_macro": f1m}
EOF

# -------------------------------
# src/evaluation/plots.py
# -------------------------------
cat > src/evaluation/plots.py <<'EOF'
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion(cm, labels, out_path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), va='center', ha='center')
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()
EOF

# -------------------------------
# src/explain/lime_shap.py
# -------------------------------
cat > src/explain/lime_shap.py <<'EOF'
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime.lime_text import LimeTextExplainer

def explain_with_lime(vectorizer, clf, texts, out_dir, topk=10, class_names=("neg","pos")):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    explainer = LimeTextExplainer(class_names=list(class_names))
    for i, t in enumerate(texts):
        exp = explainer.explain_instance(
            t,
            classifier_fn=lambda xs: clf.predict_proba(vectorizer.transform(xs)),
            num_features=topk
        )
        html_path = out / f"lime_ex_{i}.html"
        exp.save_to_file(str(html_path))

def explain_with_shap(vectorizer, clf, texts, out_dir, topk=10):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # KernelExplainer sobre probabilidades (binario)
    f = lambda xs: clf.predict_proba(vectorizer.transform(xs))[:,1]  # prob POS
    background = np.array([""])  # mínimo
    explainer = shap.KernelExplainer(f, background)
    for i, t in enumerate(texts):
        sv = explainer.shap_values([t], nsamples=100)
        # bar plot con valores absolutos (importancia)
        vals = np.abs(sv).ravel()
        # SHAP text plot (si hay tokenizer); aquí usamos bar sencillo
        idx = np.argsort(vals)[-topk:]
        plt.figure(figsize=(7,4))
        plt.barh(range(len(idx)), vals[idx])
        plt.yticks(range(len(idx)), [f"feat_{k}" for k in idx])
        plt.title(f"SHAP — Ej {i}")
        plt.tight_layout()
        plt.savefig(out / f"shap_ex_{i}.png", dpi=160)
        plt.close()
EOF

# -------------------------------
# scripts/main_train.py
# -------------------------------
cat > scripts/main_train.py <<'EOF'
#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from src.utils.io import load_yaml, save_json, ensure_dir
from src.data.datasets import load_dataset
from src.features.text import build_vectorizer
from src.models.train import split_xy, get_model, evaluate
from src.evaluation.plots import plot_confusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_default.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    df, text_col, label_col = load_dataset(cfg)
    X_train, X_test, y_train, y_test = split_xy(
        df, text_col, label_col,
        test_size=cfg["training"]["test_size"],
        stratify=cfg["training"]["stratify"],
        seed=cfg["project"]["seed"]
    )

    vec = build_vectorizer(cfg)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = get_model(cfg)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)

    metrics = evaluate(y_test, y_pred, labels=("neg","pos"))
    ensure_dir(cfg["project"]["output_dir"])
    save_json(metrics, Path(cfg["project"]["output_dir"]) / "metrics_cls.json")
    plot_confusion(metrics["confusion"], ["neg","pos"], Path(cfg["project"]["figures_dir"]) / "confusion_matrix.png")

    # Guardados simples (pickles opcionales)
    from joblib import dump
    ensure_dir("models")
    dump(vec, "models/vectorizer.joblib")
    dump(clf, "models/model.joblib")

    print("OK — entrenamiento listo. Métricas en reports/, artefactos en models/")

if __name__ == "__main__":
    main()
EOF
chmod +x scripts/main_train.py

# -------------------------------
# scripts/run_explain.py
# -------------------------------
cat > scripts/run_explain.py <<'EOF'
#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from joblib import load
from src.utils.io import load_yaml, ensure_dir
from src.data.datasets import load_dataset
from src.explain.lime_shap import explain_with_lime, explain_with_shap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_default.yaml")
    parser.add_argument("--method", choices=["lime","shap","both"], default="both")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    df, text_col, label_col = load_dataset(cfg)

    vec = load("models/vectorizer.joblib")
    clf = load("models/model.joblib")

    # Elegimos muestras de test/simple: aleatorias
    np.random.seed(cfg["project"]["seed"])
    texts = df[text_col].astype(str).sample(n=args.samples, random_state=cfg["project"]["seed"]).tolist()

    out_dir = Path(cfg["project"]["explain_dir"])
    if args.method in ("lime","both"):
        explain_with_lime(vec, clf, texts, out_dir, topk=cfg["explain"]["topk"], class_names=("neg","pos"))
    if args.method in ("shap","both"):
        explain_with_shap(vec, clf, texts, out_dir, topk=cfg["explain"]["topk"])

    print(f"OK — explicaciones guardadas en {out_dir}")

if __name__ == "__main__":
    main()
EOF
chmod +x scripts/run_explain.py

# -------------------------------
# scripts/init_dataset.py (opcional: exportar CSV a data/raw si se quiere)
# -------------------------------
cat > scripts/init_dataset.py <<'EOF'
#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from src.utils.io import load_yaml
from src.data.datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_default.yaml")
    parser.add_argument("--export_csv", action="store_true", help="Guarda una copia en data/raw/opiniones.csv")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    df, text_col, label_col = load_dataset(cfg)
    print(df.head())

    if args.export_csv:
        out = Path("data/raw/opiniones.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print("CSV exportado a", out)

if __name__ == "__main__":
    main()
EOF
chmod +x scripts/init_dataset.py

# -------------------------------
# README.md
# -------------------------------
cat > README.md <<'EOF'
# Interpretando Modelos de Clasificación de Opiniones con LIME y SHAP

**Objetivo**: Entrenar un clasificador binario (opiniones positivo/negativo) y explicar sus predicciones con **LIME** y **SHAP**.  
Proyecto modular, inspirado en ejercicios anteriores.

## Estructura
