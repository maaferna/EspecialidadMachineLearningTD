import sys
from pathlib import Path
import matplotlib.pyplot as plt

# 📌 Añadir src/ al path correctamente desde scripts/
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.utils import cargar_dataset, preprocesar_datos, entrenar_modelo_base
from src.optimizacion import (
    optimizar_con_gridsearch,
    optimizar_con_randomsearch,
    optimizar_con_optuna,
)
from src.visualizador import (
    visualizar_matriz_confusion,
    visualizar_curva_roc,
    graficar_metricas_comparativas,
)

if __name__ == "__main__":
    print("\n🚀 Iniciando pipeline de clasificación de diabetes...\n")
    df = cargar_dataset()
    X_train, X_test, y_train, y_test = preprocesar_datos(df)

    resultados = []

    # Modelo base
    resultado_base = entrenar_modelo_base(X_train, y_train, X_test, y_test)
    resultados.append({
        "metodo": "Base",
        "f1": resultado_base["f1"],
        "precision": resultado_base["precision"],
        "recall": resultado_base["recall"],
        "auc": resultado_base["auc"]
    })
    visualizar_matriz_confusion(y_test, resultado_base["y_pred"], metodo="Base")
    visualizar_curva_roc(y_test, resultado_base["y_prob"], metodo="Base")

    # GridSearch
    resultado_grid = optimizar_con_gridsearch(X_train, y_train, X_test, y_test)
    resultados.append({
        "metodo": "GridSearch",
        "f1": resultado_grid["f1"],
        "precision": resultado_grid["precision"],
        "recall": resultado_grid["recall"],
        "auc": resultado_grid["auc"]
    })
    visualizar_matriz_confusion(y_test, resultado_grid["y_pred"], metodo="GridSearch")
    visualizar_curva_roc(y_test, resultado_grid["y_prob"], metodo="GridSearch")

    # RandomSearch
    resultado_random = optimizar_con_randomsearch(X_train, y_train, X_test, y_test)
    resultados.append({
        "metodo": "RandomSearch",
        "f1": resultado_random["f1"],
        "precision": resultado_random["precision"],
        "recall": resultado_random["recall"],
        "auc": resultado_random["auc"]
    })
    visualizar_matriz_confusion(y_test, resultado_random["y_pred"], metodo="RandomSearch")
    visualizar_curva_roc(y_test, resultado_random["y_prob"], metodo="RandomSearch")

    # Optuna
    resultado_optuna = optimizar_con_optuna(X_train, y_train, X_test, y_test)
    resultados.append({
        "metodo": "Optuna",
        "f1": resultado_optuna["f1"],
        "precision": resultado_optuna["precision"],
        "recall": resultado_optuna["recall"],
        "auc": resultado_optuna["auc"]
    })
    visualizar_matriz_confusion(y_test, resultado_optuna["y_pred"], metodo="Optuna")
    visualizar_curva_roc(y_test, resultado_optuna["y_prob"], metodo="Optuna")

    # Comparación final
    graficar_metricas_comparativas(resultados)
    print("\n✅ Pipeline completado.")
