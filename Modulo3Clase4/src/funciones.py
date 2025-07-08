from sympy import symbols

def definir_funcion():
    """
    Define la función simbólica g(x, y) = x^2 + 3y^2 - 4x + 2y + 1

    Retorna:
        g (Expr): expresión simbólica de g(x, y)
        x (Symbol): variable simbólica x
        y (Symbol): variable simbólica y
    """
    x, y = symbols('x y')
    '''
    SymPy crea dos objetos de tipo Symbol, es decir, dos instancias de la clase Symbol, 
    que internamente representan variables simbólicas matemáticas:

    x → instancia que representa la variable simbólica “x”

    y → instancia que representa la variable simbólica “y”
    '''
    g = x**2 + 3*y**2 - 4*x + 2*y + 1
    print(f"Definiendo la función g(x, y): {g}")
    print(f"Definiendo las variables simbólicas: x = {x}, y = {y}")
    return g, x, y