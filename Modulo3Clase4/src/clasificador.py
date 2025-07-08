def clasificar_punto_critico(hess, punto):
    """
    Clasifica el punto crítico según los valores propios de la Hessiana.

    Args:
        hess (Matrix): Matriz Hessiana simbólica (2x2)
        punto (dict): Diccionario con los valores {x: valor, y: valor}

    Retorna:
        tipo (str): descripción del tipo de punto crítico
    """

    # Evalúa la matriz Hessiana en el punto crítico
    h_eval = hess.subs(punto)
    print(f"Evaluando la Hessiana en el punto crítico: {punto}")

    # Calcula los valores propios (eigenvalores) de la matriz evaluada
    valores_propios = h_eval.eigenvals()
    print(f"Valores propios de la Hessiana: {valores_propios}")

    # Extrae solo los valores propios (las claves del diccionario)
    vals = list(valores_propios.keys())
    print(f"Valores propios extraídos: {vals}")

    if all(val > 0 for val in vals):
        return "Mínimo local"
    elif all(val < 0 for val in vals):
        return "Máximo local"
    else:
        return "Punto de silla"
