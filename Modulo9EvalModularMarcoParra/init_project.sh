#!/usr/bin/env bash
set -euo pipefail

# Carpetas (en la carpeta actual)
mkdir -p src/data src/models src/utils scripts data/raw data/processed models outputs images reports

# .gitignore básico (opcional)
if [ ! -f .gitignore ]; then
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.venv/
.env
.DS_Store
.vscode/
.ipynb_checkpoints/
data/
models/
outputs/
images/
reports/
EOF
fi

############################################
#  src/data: carga y limpieza mínima
############################################
cat > src/data/load_heart.py << 'EOF'
from __future__ import annotations
import pandas as pd
from pathlib import Path

"""
Carga el dataset Heart Failure Prediction de Kaggle:
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Coloca el CSV en: data/raw/heart.csv
(Archivo típico: heart.csv con columnas como: Age, Sex, ChestPainType, ... , HeartDisease)

El script hace:
- Lectura del CSV
- Limpieza mínima (dropna)
- Casting básico de tipos
- Separación X, y (y = 'HeartDisease' o 'target' según dataset)
"""

RAW_PATH = Path("data/raw/heart.csv")

def load_heart_dataframe() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {RAW_PATH}. Descarga el CSV de Kaggle y guárdalo ahí."
        )
    df = pd.read_csv(RAW_PATH)
    # Normalizar nombre de la columna target
    target_candidates = ["HeartDisease", "target", "Outcome"]
    target = None
    for c in target_candidates:
        if c in df.columns:
            target = c
            break
    if target is None:
        raise ValueError("No se encontró columna de target (HeartDisease/target/Outcome) en el CSV.")

    # Limpieza mínima
    df = df.dropna().copy()

    # Casting básico (ejemplo: convertir 'Sex' y otras categóricas a category si existen)
    for col in df.columns:
        if df[col].dtype == object and col != target:
            df[col] = df[col].astype("category")

    return df, target
EOF

############################################
#  src/models: entrenamiento + persistencia
############################################
cat > src/models/train.py << 'EOF'
from __future__ import annotations
import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data.load_heart import load_heart_dataframe

MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR = Path("outputs"); OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

ARTIFACT_PATH = MODELS_DIR / "model_rf.joblib"
REPORT_TXT = OUTPUTS_DIR / "metrics.txt"

def train_test_evaluate(random_state: int = 42):
    df, target = load_heart_dataframe()

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Identificar numéricas y categóricas
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary" if y.nunique()==2 else "macro")

    # Persistir artefacto con metadatos
    joblib.dump({
        "model": pipe,
        "target": target,
        "feature_names": list(X.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }, ARTIFACT_PATH)

    # Guardar métricas
    with REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write("== Métricas de evaluación (test) ==\n")
        f.write(f"Accuracy: {acc:.3f}\n")
        f.write(f"F1-score: {f1:.3f}\n\n")
        f.write("== Classification report ==\n")
        f.write(classification_report(y_test, y_pred, digits=3))

    print(f"Modelo guardado en: {ARTIFACT_PATH}")
    print(f"Métricas guardadas en: {REPORT_TXT}")
    return ARTIFACT_PATH, REPORT_TXT

if __name__ == "__main__":
    train_test_evaluate()
EOF

############################################
#  scripts: evaluar (impresión rápida)
############################################
cat > scripts/evaluate.py << 'EOF'
from pathlib import Path
print((Path("outputs")/"metrics.txt").read_text(encoding="utf-8"))
EOF

############################################
#  scripts: SHAP (global + 3 casos individuales)
############################################
cat > scripts/explain_shap.py << 'EOF'
from __future__ import annotations
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.data.load_heart import load_heart_dataframe

OUTPUTS = Path("outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
IMAGES = Path("images"); IMAGES.mkdir(parents=True, exist_ok=True)
ETHICS = Path("reports"); ETHICS.mkdir(parents=True, exist_ok=True)

ARTIFACT_PATH = Path("models/model_rf.joblib")

def main(random_state: int = 42, n_examples: int = 3):
    # Cargar datos y modelo
    df, target = load_heart_dataframe()
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # train/test split (para seleccionar casos)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    artifact = joblib.load(ARTIFACT_PATH)
    pipe = artifact["model"]

    # Sacar las columnas transformadas para SHAP
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    X_train_enc = pre.fit_transform(X_train)  # asegurar mismo fitting
    feature_names = pre.get_feature_names_out()

    # Usar Explainer genérico (compatible con pipelines)
    explainer = shap.Explainer(pipe.predict_proba, pre.transform(X), feature_names=feature_names)
    shap_values = explainer(X_test)

    # Global — summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values[...,1], X_test[feature_names] if isinstance(X_test, pd.DataFrame) else shap_values.data, show=False)
    plt.tight_layout()
    plt.savefig(IMAGES/"shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Global — bar plot (importancias promedio absolutas)
    plt.figure()
    shap.summary_plot(shap_values[...,1], X_test[feature_names] if isinstance(X_test, pd.DataFrame) else shap_values.data, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(IMAGES/"shap_summary_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Individual — waterfall para n_examples casos
    idxs = np.random.RandomState(random_state).choice(len(X_test), size=min(n_examples, len(X_test)), replace=False)
    for i, idx in enumerate(idxs, 1):
        plt.figure()
        shap.plots.waterfall(shap_values[idx,1], show=False)
        plt.tight_layout()
        plt.savefig(IMAGES/f"shap_waterfall_case_{i}.png", dpi=160, bbox_inches="tight")
        plt.close()

    # Ética/sesgo: reporte simple con top variables globales
    # (si existe 'Sex'/'sex' se marca como potencial variable sensible)
    top = np.abs(shap_values[...,1].values).mean(axis=0)
    order = np.argsort(top)[::-1]
    ordered_names = np.array(feature_names)[order]
    ordered_vals = top[order]
    sens_flags = []
    for name, val in zip(ordered_names[:20], ordered_vals[:20]):
        sens = "SENSIBLE" if name.lower().startswith("sex") else ""
        sens_flags.append((name, float(val), sens))

    with (ETHICS/"ethics_bias_shap.md").open("w", encoding="utf-8") as f:
        f.write("# Análisis de sesgo (SHAP — global)\n\n")
        f.write("| feature | mean(|SHAP|) | flag |\n|---|---:|---|\n")
        for name, val, sens in sens_flags:
            f.write(f"| {name} | {val:.6f} | {sens} |\n")
        f.write("\n*Nota:* Revise variables potencialmente sensibles (ej. `Sex`) y considere mitigación.\n")

    print("SHAP: gráficos guardados en images/, y reporte en reports/ethics_bias_shap.md")

if __name__ == "__main__":
    main()
EOF

############################################
#  scripts: LIME (mismos casos que SHAP)
############################################
cat > scripts/explain_lime.py << 'EOF'
from __future__ import annotations
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from src.data.load_heart import load_heart_dataframe

IMAGES = Path("images"); IMAGES.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH = Path("models/model_rf.joblib")

def main(random_state: int = 42, n_examples: int = 3):
    df, target = load_heart_dataframe()
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # train/test split (coherente con SHAP)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    artifact = joblib.load(ARTIFACT_PATH)
    pipe = artifact["model"]
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # Para LIME necesitamos arrays + nombres de features transformadas
    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)
    feature_names = pre.get_feature_names_out()

    class_names = [str(c) for c in sorted(y.unique())]

    explainer = LimeTabularExplainer(
        training_data=X_train_enc.toarray() if hasattr(X_train_enc, "toarray") else X_train_enc,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        random_state=random_state
    )

    idxs = np.random.RandomState(random_state).choice(len(X_test_enc), size=min(n_examples, len(X_test_enc)), replace=False)

    for i, idx in enumerate(idxs, 1):
        x = X_test_enc[idx]
        predict_fn = lambda data: pipe.predict_proba(pre.inverse_transform(data))
        exp = explainer.explain_instance(
            data_row=x.toarray()[0] if hasattr(x, "toarray") else x,
            predict_fn=predict_fn,
            num_features=10
        )
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(IMAGES/f"lime_explanation_case_{i}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print("LIME: figuras guardadas en images/ (mismos casos que SHAP según random_state)")

if __name__ == "__main__":
    main()
EOF

############################################
#  scripts: reporte final (Markdown consolidado)
############################################
cat > scripts/make_report.py << 'EOF'
from pathlib import Path

OUT = Path("reports"); OUT.mkdir(parents=True, exist_ok=True)

def section(title: str) -> str:
    return f"\\n\\n## {title}\\n\\n"

def main():
    md = ["# Informe — Interpretabilidad con LIME & SHAP"]

    # 1. Carga y exploración (instrucciones)
    md.append(section("1) Carga y exploración de datos"))
    md.append("- Dataset: Heart Failure Prediction (Kaggle).")
    md.append("- Limpieza mínima: `dropna`, casting categóricas.")
    md.append("- Variables potencialmente sensibles: `Sex` (u otras relacionadas).")

    # 2. Modelo
    md.append(section("2) Construcción y evaluación del modelo"))
    metrics_path = Path("outputs/metrics.txt")
    if metrics_path.exists():
        md.append("**Métricas (test):**\n")
        md.append("```")
        md.append(metrics_path.read_text(encoding="utf-8"))
        md.append("```")
    else:
        md.append("_Ejecuta `python -m src.models.train` para generar métricas._")

    # 3. SHAP
    md.append(section("3) SHAP — Explicaciones"))
    md.append("Gráficos globales y locales:")
    for name in ["shap_summary_beeswarm.png", "shap_summary_bar.png", "shap_waterfall_case_1.png", "shap_waterfall_case_2.png", "shap_waterfall_case_3.png"]:
        p = Path("images")/name
        if p.exists(): md.append(f"![{name}](../images/{name})")

    eb = Path("reports/ethics_bias_shap.md")
    if eb.exists():
        md.append("\n**Análisis de posibles sesgos (SHAP):**\n")
        md.append(eb.read_text(encoding="utf-8"))

    # 4. LIME
    md.append(section("4) LIME — Explicaciones locales"))
    for name in ["lime_explanation_case_1.png", "lime_explanation_case_2.png", "lime_explanation_case_3.png"]:
        p = Path("images")/name
        if p.exists(): md.append(f"![{name}](../images/{name})")

    # 5. Sesgo y ética
    md.append(section("5) Análisis de sesgo y ética"))
    md.append("- Revisar si `Sex` (u otra variable sensible) tiene peso desproporcionado.")
    md.append("- Mitigación potencial: balanceo, eliminación o anonimización de variables, regularización, ajuste de umbral, post-procesamiento de fairness.")
    md.append("- Riesgo sin interpretabilidad: decisiones clínicas injustas o no auditables.")

    # 6. Propuesta de mejora
    md.append(section("6) Propuesta de mejora"))
    md.append("- Evaluar otro algoritmo (e.g., XGBoost) y comparar métricas + explicaciones.")
    md.append("- Ajustar preprocesamiento (codificación, estandarización, manejo de outliers).")
    md.append("- Re-entrenar sin variables sensibles y comparar desempeño/explicaciones.")

    # Guardar
    out_path = OUT/"informe_interpretabilidad.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Reporte consolidado: {out_path}")

if __name__ == "__main__":
    main()
EOF

# __init__ para imports relativos
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/utils/__init__.py

echo "✅ Estructura y archivos de interpretabilidad creados/actualizados en la carpeta actual."
