from src.utils import cargar_y_preprocesar_credit
from src.modelos import entrenar_logistic_regression, entrenar_random_forest

def main():
    print("🔹 Iniciando pipeline para Credit Scoring")

    X_train, X_test, y_train, y_test = cargar_y_preprocesar_credit()

    # Logistic Regression
    resultados_logreg = entrenar_logistic_regression(X_train, y_train, X_test, y_test)

    # Random Forest
    resultados_rf = entrenar_random_forest(X_train, y_train, X_test, y_test)

    print("\n📊 Comparación final:")
    print("Logistic Regression:", resultados_logreg["resultados"])
    print("Random Forest:", resultados_rf["resultados"])

  

if __name__ == "__main__":
    main()
