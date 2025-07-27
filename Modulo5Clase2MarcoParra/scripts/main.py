
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import (
    cargar_y_preprocesar_california,
    cargar_y_preprocesar_ingresos,
    cargar_y_preprocesar_macro,
)
from src.modelos import (
    entrenar_elasticnet,
    entrenar_regresion_cuantilica,
    entrenar_var,
)

from src.visualizador import (
    graficar_coeficientes_elasticnet,
    graficar_pinball_loss_quantiles,
    graficar_forecast_var,
)

warnings.filterwarnings("ignore")

# Grid de parámetros
grid_params = {
    "elasticnet": [
        {"alpha": 0.001, "l1_ratio": 0.1},
        {"alpha": 0.01, "l1_ratio": 0.5},
        {"alpha": 0.1, "l1_ratio": 0.7},
        {"alpha": 1.0, "l1_ratio": 0.9},
    ],
    "quantiles": list(map(lambda x: round(x, 2), list(np.arange(0.1, 1.0, 0.1)))),
    "var": {"maxlags": 10}
}

def main():
    print("\n🔹 1. Dataset: California Housing")
    X_cali, y_cali = cargar_y_preprocesar_california()
    print(f"✅ California Housing -> X: {X_cali.shape}, y: {y_cali.shape}")

    # Train/Test split
    X_train_cali, X_test_cali, y_train_cali, y_test_cali = train_test_split(
        X_cali, y_cali, test_size=0.2, random_state=42
    )

    print("\n🔹 Entrenando modelo ElasticNet...")
    model_elastic = entrenar_elasticnet(X_train_cali, y_train_cali, X_test_cali, y_test_cali, grid_params["elasticnet"])
    graficar_coeficientes_elasticnet(model_elastic, X_train_cali.columns)    
    print("\n🔹 Visualizando coeficientes del modelo ElasticNet...")
    graficar_coeficientes_elasticnet(model_elastic, X_train_cali.columns)

    print("\n🔹 2. Dataset: Adult Income (para cuantílica)")
    X_income, y_income = cargar_y_preprocesar_ingresos()
    print(f"✅ Adult Income -> X: {X_income.shape}, y: {y_income.shape}")

    # Train/Test split
    X_train_income, X_test_income, y_train_income, y_test_income = train_test_split(
        X_income, y_income, test_size=0.2, random_state=42
    )

    print("\n🔹 Entrenando modelo de Regresión Cuantílica...")
    entrenar_regresion_cuantilica(X_train_income, y_train_income, X_test_income, y_test_income, grid_params["quantiles"])
    print("\n🔹 Visualizando Pinball Loss para cuantiles...")
    graficar_pinball_loss_quantiles()

    print("\n🔹 3. Dataset: Indicadores macroeconómicos (VAR)")
    df_macro = cargar_y_preprocesar_macro()
    print(f"✅ Macro dataset -> shape: {df_macro.shape}")

    print("\n🔹 Entrenando modelo VAR...")
    entrenar_var(df_macro, grid_params["var"])
    print("\n🔹 Visualizando forecast VAR...")
    graficar_forecast_var()

    print("\n✅ Proceso completado exitosamente.")

if __name__ == "__main__":
    main()