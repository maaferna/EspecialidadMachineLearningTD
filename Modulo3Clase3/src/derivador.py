from sympy import symbols, diff, Eq, solve

def derivar_funcion(f):
    """
    Calcula la derivada simbólica de una función f(x).

    Args:
        f (Expr): expresión simbólica de f(x)

    Retorna:
        tuple: (f_expr, df_expr, x_symbol)
    """
    x = symbols('x')  # Define un símbolo simbólico 'x' usando SymPy para manipular expresiones algebraicas.
    print(f"Definiendo la variable simbólica: {x}")
    print(f"Definiendo la función f(x): {f}")
    
    df = diff(f, x)    # Calcula la derivada simbólica de f con respecto a x: df = f'(x) = 2(x - 3).
    print(f"Calculando la derivada simbólica: f'(x) = {df}")
    
    f_expr = f.simplify()  # Simplifica la expresión de f para una mejor legibilidad.
    print(f"Expresión simplificada de f(x): {f_expr}")
    return f_expr, df, x

def encontrar_critico(df_expr, x):
    """
    Resuelve f'(x) = 0 para encontrar el punto crítico.

    Args:
        df_expr (Expr): expresión de la derivada
        x (Symbol): símbolo de la variable independiente

    Retorna:
        float: valor del punto crítico
    """
    critico = solve(Eq(df_expr, 0), x)[0]
    return critico
