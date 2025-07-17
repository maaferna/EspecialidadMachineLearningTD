# src/visualizador.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay
)

def visualizar_matriz_confusion(y_true, y_pred, metodo="Modelo"):
    """
    Genera y guarda una matriz de confusión como imagen.

    Args:
        y_true: Valores verdaderos.
        y_pred: Predicciones del modelo.
        metodo: Nombre del modelo.
    """
    if y_pred is None:
        print(f"⚠️ No se puede graficar matriz de confusión para {metodo}: y_pred es None.")
        return

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusión - {metodo}")
    plt.tight_layout()
    plt.savefig(f"outputs/matriz_confusion_{metodo.lower()}.png")
    plt.show()  # <-- muestra el gráfico en el notebook

    plt.close()

def visualizar_curva_roc(y_true, y_prob, metodo="Modelo"):
    """
    Genera y guarda la curva ROC como imagen.

    Args:
        y_true: Valores verdaderos.
        y_prob: Probabilidades predichas.
        metodo: Nombre del modelo.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"Curva ROC - {metodo}")
    plt.tight_layout()
    plt.savefig(f"outputs/roc_{metodo.lower()}.png")
    plt.show()  # <-- muestra el gráfico en el notebook

    plt.close()

def graficar_metricas_comparativas_subplots(resultados_clf, resultados_reg):
    """
    Genera gráfico comparativo con subplots para clasificación y regresión.

    Args:
        resultados_clf: Lista de dicts con resultados de clasificación.
        resultados_reg: Lista de dicts con resultados de regresión.
    """
    df_clf = pd.DataFrame(resultados_clf)
    df_reg = pd.DataFrame(resultados_reg)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(data=df_clf, x="modelo", y="score", hue="modelo", ax=axes[0], palette="Blues", legend=False)
    axes[0].set_title("Modelos de Clasificación")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].bar_label(axes[0].containers[0])

    # Agrupar por modelo para mostrar solo el mejor alpha por quantil
    df_reg_grouped = df_reg.copy()
    if "parametros" in df_reg.columns:
        df_reg_grouped["quantil"] = df_reg_grouped["parametros"].apply(lambda d: d.get("quantil") if isinstance(d, dict) else None)
        df_reg_grouped["alpha"] = df_reg_grouped["parametros"].apply(lambda d: d.get("alpha") if isinstance(d, dict) else None)
        df_reg_grouped = df_reg_grouped.sort_values("score").drop_duplicates(subset=["modelo"], keep="first")

    sns.barplot(data=df_reg_grouped, x="modelo", y="score", hue="modelo", ax=axes[1], palette="Greens", legend=False)
    axes[1].set_title("Modelos de Regresión")
    axes[1].set_ylabel("Score (RMSE o Pinball Loss)")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].bar_label(axes[1].containers[0])

    plt.tight_layout()
    plt.savefig("outputs/comparacion_metricas_modelos.png")
    plt.show()  # <-- muestra el gráfico en el notebook

    plt.close()

def visualizar_pred_vs_real(y_true, y_pred, metodo="Modelo"):
    """
    Grafica y guarda una comparación entre y_real vs y_predicha.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
    plt.xlabel("Valor Real")
    plt.ylabel("Predicción")
    plt.title(f"Predicción vs Real - {metodo}")
    plt.tight_layout()
    plt.savefig(f"outputs/pred_vs_real_{metodo.lower()}.png")
    plt.show()  # <-- muestra el gráfico en el notebook

    plt.close()


# Dentro de visualizador.py

def graficar_mejores_metricas_modelos_regresion(resultados_reg):

    df = pd.DataFrame(resultados_reg)

    # Normaliza cada grupo de métricas por tipo
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=df,
        x="modelo",
        y="score",
        hue="modelo",
        palette="Greens",
        ax=ax,
        dodge=False
    )

    ax.set_title("Comparación de Métricas de Modelos de Regresión")
    ax.set_ylabel("RMSE / Pinball Loss (según modelo)")
    ax.set_xlabel("Modelo")
    ax.tick_params(axis='x', rotation=45)
    ax.bar_label(ax.containers[0])

    plt.tight_layout()
    plt.savefig("outputs/comparacion_regresion_mejores.png")
    plt.show()  # <-- muestra el gráfico en el notebook

    plt.close()


def graficar_dispersion_cuantiles(resultados_reg):
    '''
    graficar_dispersion_cuantiles
    Grafica la dispersión de los resultados de regresión cuantílica por quantil y alpha.
    Args:
        resultados_reg: Lista de dicts con resultados de regresión.
    '''
    df = pd.DataFrame(resultados_reg)
    df_qr = df[df["modelo"].str.contains("QuantileRegressor")].copy()

    df_qr["quantil"] = df_qr["parametros"].apply(lambda d: d["quantil"])
    df_qr["alpha"] = df_qr["parametros"].apply(lambda d: d["alpha"])

    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=df_qr,
        x="quantil",
        y="score",
        hue="alpha",
        palette="coolwarm",
        s=100
    )

    plt.title("Dispersión de Pinball Loss por Quantil y Alpha")
    plt.xlabel("Quantil")
    plt.ylabel("Pinball Loss")
    plt.legend(title="Alpha")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/dispersion_cuantiles.png")
    plt.show()  # <-- muestra el gráfico en el notebook

    plt.close()
