from sympy import diff, Matrix, Eq, solve


def calcular_gradiente(g, x, y):
    """
    Calcula el gradiente de la función g(x, y).

    Retorna:
        grad (Matrix): vector gradiente [dg/dx, dg/dy]
    """
    dg_dx = diff(g, x)
    dg_dy = diff(g, y)
    return Matrix([dg_dx, dg_dy])


def calcular_hessiana(g, x, y):
    """
    Construye la matriz Hessiana de la función g(x, y).

    Retorna:
        hess (Matrix): matriz Hessiana 2x2
    """
    hxx = diff(g, x, x)
    hxy = diff(g, x, y)
    hyx = diff(g, y, x)
    hyy = diff(g, y, y)
    return Matrix([[hxx, hxy], [hyx, hyy]])


def encontrar_punto_critico(grad, x, y):
    """
    Resuelve el sistema gradiente = (0, 0).

    Retorna:
        soluciones (list): lista de puntos críticos
    """
    # Resuelve un sistema de ecuaciones simbólicas definido por igualar cada componente del gradiente a cero
    soluciones = solve([Eq(grad[0], 0), Eq(grad[1], 0)], (x, y), dict=True)
    return soluciones


