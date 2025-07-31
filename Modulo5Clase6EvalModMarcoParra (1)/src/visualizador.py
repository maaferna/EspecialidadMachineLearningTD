# src/visualizador.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc

def graficar_matriz_confusion(cm, modelo, output_dir="outputs"):
    """Genera y guarda la matriz de confusión."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    plt.title(f"Matriz de Confusión - {modelo}")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{modelo.lower()}.png"))
    plt.close()


def graficar_curva_roc(best_model, X_test, y_test, modelo, output_dir="outputs"):
    """Genera y guarda la curva ROC si el modelo soporta predict_proba."""
    if not hasattr(best_model, "predict_proba"):
        print(f"⚠️ {modelo} no soporta predict_proba, se omite la curva ROC.")
        return

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"Curva ROC - {modelo}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"roc_curve_{modelo.lower()}.png"))
    plt.close()


def graficar_importancia_variables(best_model, X_train, modelo, output_dir="outputs"):
    """Genera gráfico de importancia de variables o coeficientes."""
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)[:15]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
        plt.title(f"Importancia de Variables - {modelo}")
        plt.xlabel("Importancia")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_importance_{modelo.lower()}.png"))
        plt.close()

    elif hasattr(best_model, "coef_"):
        coef = pd.Series(best_model.coef_.flatten(), index=X_train.columns)
        coef = coef.sort_values(key=abs, ascending=False)[:15]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=coef.values, y=coef.index, palette="magma")
        plt.title(f"Coeficientes - {modelo}")
        plt.xlabel("Peso del coeficiente")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"coeficientes_{modelo.lower()}.png"))
        plt.close()
    else:
        print(f"⚠️ {modelo} no tiene atributos de importancia de variables.")
