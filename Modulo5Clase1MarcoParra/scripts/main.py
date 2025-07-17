# main.py

import pandas as pd
from src.utils import (
    cargar_dataset,
    preprocesar_datos_clasificacion,
    preprocesar_datos_regresion,
)
from src.entrenamiento import (
    entrenar_random_forest,
    entrenar_xgboost,
    entrenar_elasticnet,
    entrenar_regresion_cuantil,
)
from src.visualizador import (
    graficar_dispersion_cuantiles,
    graficar_mejores_metricas_modelos_regresion,
    visualizar_matriz_confusion,
    visualizar_curva_roc,
    graficar_metricas_comparativas_subplots,
    visualizar_pred_vs_real,
)

# ======================
# Carga y preprocesamiento
# ======================

df = cargar_dataset()

# Clasificación
X_train_clf, X_test_clf, y_train_clf, y_test_clf, _, _ = preprocesar_datos_clasificacion(df)
# Regresión
X_train_reg, X_test_reg, y_train_reg, y_test_reg, _ = preprocesar_datos_regresion(df)

# ======================
# Entrenamiento y Evaluación
# ======================

resultados_clf = []
resultados_reg = []

# Random Forest (Clasificación)
rf_clf = entrenar_random_forest(X_train_clf, y_train_clf, X_test_clf, y_test_clf)
resultados_clf.append(rf_clf)
visualizar_matriz_confusion(y_test_clf, rf_clf.get("y_pred"), metodo="RandomForest")
visualizar_curva_roc(y_test_clf, rf_clf.get("y_prob"), metodo="RandomForest")

# XGBoost (Clasificación)
xgb_clf = entrenar_xgboost(X_train_clf, y_train_clf, X_test_clf, y_test_clf)
resultados_clf.append(xgb_clf)
visualizar_matriz_confusion(y_test_clf, xgb_clf.get("y_pred"), metodo="XGBoost")
visualizar_curva_roc(y_test_clf, xgb_clf.get("y_prob"), metodo="XGBoost")

# ElasticNet (Regresión)
elastic_reg = entrenar_elasticnet(X_train_reg, y_train_reg, X_test_reg, y_test_reg, n_iter=30)
resultados_reg.append(elastic_reg)
visualizar_pred_vs_real(y_test_reg, elastic_reg.get("y_pred"), metodo="ElasticNet")

# Regresión Cuantílica (0.1, 0.5, 0.9)
cuantil_resultados = entrenar_regresion_cuantil(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
resultados_reg.extend(cuantil_resultados)
# ======================
# Visualización Comparativa
# ======================

graficar_metricas_comparativas_subplots(resultados_clf, resultados_reg)

# Al final del main.py
graficar_mejores_metricas_modelos_regresion(resultados_reg)
graficar_dispersion_cuantiles(resultados_reg)


print("\n✅ Pipeline completado.")
