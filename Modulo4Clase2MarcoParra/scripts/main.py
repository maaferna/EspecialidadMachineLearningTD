import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Ajustar path para importar desde src/
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.utils import cargar_dataset, preprocesar_datos, entrenar_modelo_base
from src.optimizacion import (
    optimizar_con_gridsearch,
    optimizar_con_randomsearch,
)
from src.visualizador import (
    visualizar_matriz_confusion,
    visualizar_curva_roc,
    graficar_metricas_comparativas,
)


if __name__ == "__main__":
    print("\n🚀 Iniciando pipeline de clasificación de diabetes...\n")

    # 1. Cargar y preprocesar datos
    df = cargar_dataset()
    X_train, X_test, y_train, y_test = preprocesar_datos(df)

    resultados = []

    # 2. Entrenar modelo base
    resultado_base = entrenar_modelo_base(X_train, y_train, X_test, y_test)
    resultados.append(resultado_base)
    visualizar_matriz_confusion(y_test, resultado_base["y_pred"], metodo="Base")
    visualizar_curva_roc(y_test, resultado_base["y_prob"], metodo="Base")

    # 3. Grid Search
    resultado_grid = optimizar_con_gridsearch(X_train, y_train, X_test, y_test)
    resultados.append(resultado_grid)
    visualizar_matriz_confusion(y_test, resultado_grid["y_pred"], metodo="GridSearch")
    visualizar_curva_roc(y_test, resultado_grid["y_prob"], metodo="GridSearch")

    # 4. Random Search
    resultado_random = optimizar_con_randomsearch(X_train, y_train, X_test, y_test)
    resultados.append(resultado_random)
    visualizar_matriz_confusion(y_test, resultado_random["y_pred"], metodo="RandomSearch")
    visualizar_curva_roc(y_test, resultado_random["y_prob"], metodo="RandomSearch")

    # 5. Comparar resultados
    graficar_metricas_comparativas(resultados)

    # 6. Reflexión
    print("\n📌 Reflexión Final:")
    print("- GridSearch suele encontrar buenos modelos pero puede ser más lento.")
    print("- RandomSearch explora más y puede ser más eficiente con menos combinaciones.")
    print("- Comparar tiempos, F1 y AUC para definir cuál se adapta mejor al problema.")

    print("\n✅ Pipeline completado.")
