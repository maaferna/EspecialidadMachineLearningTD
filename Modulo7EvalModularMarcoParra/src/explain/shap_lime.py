# src/explain/shap_lime.py
import os
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

def shap_summary_kernelexplainer(model, X_train, X_sample, feature_names, out_png: str):
    """
    KernelExplainer (modelo agnóstico). Para ser rápido:
      - Usa una muestra pequeña de background (p.ej., 200)
      - Explica ~200 filas
    """
    background = shap.sample(X_train, 200, random_state=42)
    explainer = shap.KernelExplainer(model.predict, background)
    sv = explainer.shap_values(X_sample, nsamples="auto")
    # Para binario con sigmoid -> sv es una lista (clase positiva)
    vals = sv[0] if isinstance(sv, list) else sv
    shap.summary_plot(vals, X_sample, feature_names=feature_names, show=False)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def lime_explain_instances(model, X_train, X_test, feature_names, class_names, out_dir: str, num_instances: int = 3):
    explainer = LimeTabularExplainer(
        training_data=X_train, feature_names=feature_names,
        class_names=class_names, discretize_continuous=True, mode="classification"
    )
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(num_instances, len(X_test))):
        exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=10)
        html_path = os.path.join(out_dir, f"lime_instance_{i}.html")
        exp.save_to_file(html_path)
