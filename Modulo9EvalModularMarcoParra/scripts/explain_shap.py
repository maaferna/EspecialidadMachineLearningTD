from __future__ import annotations
import joblib
from pathlib import Path
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.data.load_heart import load_heart_dataframe

OUTPUTS = Path("outputs"); OUTPUTS.mkdir(parents=True, exist_ok=True)
IMAGES = Path("images"); IMAGES.mkdir(parents=True, exist_ok=True)
ETHICS = Path("reports"); ETHICS.mkdir(parents=True, exist_ok=True)

ARTIFACT_PATH = Path("models/model_rf.joblib")

def main(random_state: int = 42, n_examples: int = 3):
    """Genera explicaciones SHAP globales y locales (trabajando en espacio transformado).
    figuras en images/shap_*.png y reporte simple en reports/ethics_bias_shap.md
    Args:
        random_state (int): semilla para reproducibilidad.
        n_examples (int): número de explicaciones locales a generar.
    Returns:
        None
    """
    # 1) Datos y split
    df, target = load_heart_dataframe()
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 2) Modelo
    artifact = joblib.load(ARTIFACT_PATH)
    pipe = artifact["model"]
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # 3) Espacio transformado (denso)
    X_train_enc = pre.fit_transform(X_train)
    X_test_enc  = pre.transform(X_test)
    if hasattr(X_train_enc, "toarray"):
        X_train_enc = X_train_enc.toarray()
    if hasattr(X_test_enc, "toarray"):
        X_test_enc = X_test_enc.toarray()
    feature_names = pre.get_feature_names_out()

    # 4) TreeExplainer sobre el clasificador (RF) y features transformadas
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_enc, check_additivity=False)  # binaria -> lista [clase0, clase1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # 5) Globales
    plt.figure()
    shap.summary_plot(sv, X_test_enc, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(IMAGES/"shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, X_test_enc, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(IMAGES/"shap_summary_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    # 6) Locales (waterfall): usar expected_value escalar de la clase 1
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)):
        ev = ev[1]
    idxs = np.random.RandomState(random_state).choice(len(X_test_enc), size=min(n_examples, len(X_test_enc)), replace=False)
    for i, idx in enumerate(idxs, 1):
        plt.figure()
        shap.plots._waterfall.waterfall_legacy(ev, sv[idx], feature_names=feature_names, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(IMAGES/f"shap_waterfall_case_{i}.png", dpi=160, bbox_inches="tight")
        plt.close()

    # 7) Reporte simple de “posible sesgo”
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ordered_names = np.array(feature_names)[order]
    ordered_vals  = mean_abs[order]
    with (ETHICS/"ethics_bias_shap.md").open("w", encoding="utf-8") as f:
        f.write("# Análisis de sesgo (SHAP — global)\n\n")
        f.write("| feature | mean(|SHAP|) |\n|---|---:|\n")
        for name, val in zip(ordered_names[:20], ordered_vals[:20]):
            f.write(f"| {name} | {val:.6f} |\n")
        f.write("\n*Revisar variables potencialmente sensibles (ej. `Sex`).*\n")

    print("SHAP listo: figuras en images/ y reporte en reports/ethics_bias_shap.md")

if __name__ == "__main__":
    main()
