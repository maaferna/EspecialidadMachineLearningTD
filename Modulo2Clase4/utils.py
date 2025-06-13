# -----------------------------
# 1. 📊 Funciones de Búsqueda
# -----------------------------
def busqueda_lineal(lista, objetivo):
    """
    Realiza una búsqueda lineal del objetivo en la lista.

    Args:
        lista (list): Lista de enteros.
        objetivo (int): Valor a buscar.

    Returns:
        int: Índice del objetivo o -1 si no se encuentra.
    """
    for i, valor in enumerate(lista):
        if valor == objetivo:
            return i
    return -1

def busqueda_binaria(lista, objetivo):
    """
    Realiza una búsqueda binaria del objetivo en la lista ordenada.

    Args:
        lista (list): Lista de enteros ordenados.
        objetivo (int): Valor a buscar.

    Returns:
        int: Índice del objetivo o -1 si no se encuentra.
    """
    izquierda, derecha = 0, len(lista) - 1
    while izquierda <= derecha:
        medio = (izquierda + derecha) // 2
        if lista[medio] == objetivo:
            return medio
        elif lista[medio] < objetivo:
            izquierda = medio + 1
        else:
            derecha = medio - 1
    return -1