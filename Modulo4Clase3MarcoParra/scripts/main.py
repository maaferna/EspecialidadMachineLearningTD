"""
main.py - Entrenamiento y optimización de RandomForestClassifier
usando técnicas de Optimización Bayesiana (Scikit-Optimize y Hyperopt).
"""

import sys
from pathlib import Path

# 📌 Añadir src/ al path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from src.utils import (
    cargar_datos,
    entrenar_modelo_base,
    optimizar_con_skopt,
    optimizar_con_hyperopt,
)
from src.visualizador import mostrar_evolucion, mostrar_resultados

if __name__ == "__main__":
    print("🔍 Cargando y preparando datos...")
    X_train, X_test, y_train, y_test = cargar_datos()
    resultados = []


    print("🌲 Entrenando modelo base (sin optimización)...")
    resultado_base = entrenar_modelo_base(X_train, y_train, X_test, y_test)

    resultados.append(resultado_base)

    print("\n⚙️ Optimización con Scikit-Optimize...")
    resultado_skopt = optimizar_con_skopt(X_train, y_train, X_test, y_test)
    resultados.append(resultado_skopt)

    print("\n🔧 Optimización con Hyperopt...")
    resultado_hyperopt = optimizar_con_hyperopt(X_train, y_train, X_test, y_test)
    resultados.append(resultado_hyperopt)

    print("\n📊 Comparación de resultados:")
    mostrar_resultados(resultados)
    
    mostrar_evolucion(
        skopt_scores=resultado_skopt["evolucion"],
        hyperopt_scores=resultado_hyperopt["evolucion"],
        f1_base=resultado_base["f1"]
    )
