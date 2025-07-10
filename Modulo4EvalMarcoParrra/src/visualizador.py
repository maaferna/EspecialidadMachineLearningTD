import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import os
import seaborn as sns
import pandas as pd


def visualizar_matriz_confusion(y_true, y_pred, metodo="Modelo"):
    """Dibuja y guarda la matriz de confusión."""
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title(f"Matriz de Confusión - {metodo}")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/matriz_confusion_{metodo.lower()}.png")
    plt.show()


def visualizar_curva_roc(y_true, y_scores, metodo="Modelo"):
    """Dibuja y guarda la curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title(f"Curva ROC - {metodo}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/roc_{metodo.lower()}.png")
    plt.show()


def graficar_metricas_comparativas(resultados):
    """
    Dibuja un conjunto de gráficos de barras para comparar las métricas de los modelos.
    Cada elemento en `resultados` debe ser un diccionario con las claves: metodo, f1, precision, recall, auc
    """

    # Convertir resultados a DataFrame
    df = pd.DataFrame(resultados)

    # Reorganizar para visualización
    df_melted = df.melt(id_vars="metodo", 
                        value_vars=["f1", "precision", "recall", "auc"], 
                        var_name="Métrica", 
                        value_name="Valor")

    # Renombrar para presentación más clara
    df_melted["Métrica"] = df_melted["Métrica"].str.upper().replace({
        "F1": "F1-Score",
        "PRECISION": "Precisión",
        "RECALL": "Recall",
        "AUC": "AUC"
    })

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Gráfico de líneas
    sns.lineplot(data=df_melted, x="metodo", y="Valor", hue="Métrica", marker="o", linewidth=2.5, markersize=8)

    # Añadir etiquetas numéricas
    for i in range(df_melted.shape[0]):
        row = df_melted.iloc[i]
        plt.text(x=i % len(df["metodo"]), 
                 y=row["Valor"] + 0.01, 
                 s=f"{row['Valor']:.2f}", 
                 ha='center', fontsize=9)

    plt.title("📊 Comparación de Métricas por Modelo", fontsize=16)
    plt.ylim(0.5, 1.02)
    plt.ylabel("Valor")
    plt.xlabel("Modelo")
    plt.legend(title="Métrica", loc="lower right")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/comparacion_metricas_modelos.png")
    plt.show()
