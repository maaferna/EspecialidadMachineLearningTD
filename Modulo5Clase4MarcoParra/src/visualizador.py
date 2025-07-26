import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay,
    precision_recall_curve, PrecisionRecallDisplay
)


def plot_metricas_comparativas(resultados):
    """
    Grafica una comparación de las métricas de evaluación para cada estrategia de validación cruzada.

    Args:
        resultados (list[dict]): Lista de resultados con métricas.
    """
    df = pd.DataFrame(resultados)
    metricas = ['accuracy', 'precision', 'recall', 'f1_score']

    df_melt = df.melt(id_vars='estrategia', value_vars=metricas, var_name='métrica', value_name='valor')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x='métrica', y='valor', hue='estrategia', palette='Set2')
    plt.title('Comparación de Métricas por Estrategia de Validación Cruzada')
    plt.ylim(0, 1)
    plt.legend(title='Estrategia')
    plt.tight_layout()
    plt.savefig("outputs/comparacion_metricas_validacion.png")
    plt.show()


def plot_matriz_confusion(y_true, y_pred, nombre_modelo, nombre_validacion):
    """
    Muestra la matriz de confusión.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusión - {nombre_modelo} ({nombre_validacion})")
    plt.tight_layout()
    plt.savefig(f"outputs/matriz_confusion_{nombre_modelo}_{nombre_validacion}.png")
    plt.show()


def plot_roc_curve(y_true, y_score, nombre_modelo, nombre_validacion):
    """
    Muestra la curva ROC para un modelo.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"Curva ROC - {nombre_modelo} ({nombre_validacion})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/roc_curve_{nombre_modelo}_{nombre_validacion}.png")
    plt.show()


def plot_precision_recall(y_true, y_score, nombre_modelo, nombre_validacion):
    """
    Muestra la curva Precision-Recall.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title(f"Curva Precision-Recall - {nombre_modelo} ({nombre_validacion})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/precision_recall_{nombre_modelo}_{nombre_validacion}.png")
    plt.show()


def graficar_metricas_individuales(modelo, X_train, y_train, X_test, y_test, nombre_modelo, nombre_validacion):
    """
    Entrena el modelo y genera gráficas individuales: matriz de confusión, ROC, y Precision-Recall.
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    try:
        y_score = modelo.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_score = y_pred  # fallback para modelos sin método predict_proba

    plot_matriz_confusion(y_test, y_pred, nombre_modelo, nombre_validacion)
    plot_roc_curve(y_test, y_score, nombre_modelo, nombre_validacion)
    plot_precision_recall(y_test, y_score, nombre_modelo, nombre_validacion)
