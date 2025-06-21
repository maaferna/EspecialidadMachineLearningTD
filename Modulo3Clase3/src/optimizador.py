from scipy.optimize import minimize

def optimizar_funcion():
    """
    Optimiza f(x) = (x - 3)^2 utilizando el método de minimización de SciPy.

    Returns:
        OptimizeResult: objeto con los resultados de la optimización
    """
    # Importa la función 'minimize' de SciPy para realizar la optimización.
    # Define la función objetivo f(x) = (x - 3)^2 como una lambda.
    # Se espera que 'x' sea un arreglo unidimensional, por eso se accede con x[0].
    f = lambda x: (x[0] - 3) ** 2
    print(type(f))  # Imprime el tipo de la función para verificar que es una función lambda.
    print(f)
    # Aplica el método 'minimize' comenzando desde x0 = 0 para encontrar el mínimo de la función.
    resultado = minimize(f, x0=[0])
    print(f"Resultado de la optimización: {resultado}")

    return resultado
