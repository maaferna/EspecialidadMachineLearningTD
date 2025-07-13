import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import os
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import label_binarize
import numpy as np

def visualizar_matriz_confusion(y_true, y_pred, metodo="Modelo"):
    """Dibuja y guarda la matriz de confusión.
    Soporta clasificación multiclase.
    """
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title(f"Matriz de Confusión - {metodo}")
    plt.tight_layout()
    os.makedirs("outputs_cv", exist_ok=True)
    plt.savefig(f"outputs_cv/matriz_confusion_{metodo.lower()}.png")
    plt.show()




def visualizar_curva_roc(y_true, y_prob, metodo="Modelo", n_classes=None, label_encoder=None):
    """
    Visualiza la curva ROC para problemas multiclase usando one-vs-rest.
    Soporta label_encoder para mostrar nombres de clase.
    Parámetros:
    - y_true: Etiquetas verdaderas (array de enteros o categóricas).
    - y_prob: Probabilidades predichas por el modelo (array de shape [n_samples, n_classes]).
    - metodo: Nombre del método para mostrar en el título.
    - n_classes: Número de clases (opcional, se detecta automáticamente si no se provee).
    - label_encoder: LabelEncoder usado para transformar las etiquetas (opcional).
    """
    # Detecta automáticamente número de clases si no se provee
    if n_classes is None:
        n_classes = y_prob.shape[1]
    # Binariza el target
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # Curvas ROC por clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Curva macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(all_fpr, mean_tpr, color='navy', linestyle=':', lw=2,
             label='Macro-average ROC (AUC = {:.2f})'.format(roc_auc["macro"]))
    for i in range(n_classes):
        label = f"Clase {i}"
        if label_encoder is not None:
            try:
                label = label_encoder.inverse_transform([i])[0]
            except Exception:
                pass
        plt.plot(fpr[i], tpr[i], lw=1.5,
                 label=f'ROC {label} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title(f"Curva ROC One-vs-Rest: {metodo}")
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
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

    os.makedirs("outputs_cv", exist_ok=True)
    plt.savefig("outputs_cv/comparacion_metricas_modelos.png")
    plt.show()
