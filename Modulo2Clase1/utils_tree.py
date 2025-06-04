# -----------------------------
# 🌳 Versión básica de Árbol Binario de Clasificación de Usuarios
# Este árbol representa niveles de usuario de forma simple y estática
# -----------------------------

class NodoBasico:
    """
    Nodo básico de un árbol binario con tipo de usuario.
    """
    def __init__(self, tipo):
        self.tipo = tipo
        self.izquierda = None
        self.derecha = None



# -----------------------------
# Función simple para mostrar los nodos (recorrido en orden)
# -----------------------------
def imprimir_arbol_simple(nodo):
    if nodo:
        imprimir_arbol_simple(nodo.izquierda)
        print(f"🧑 Tipo de usuario: {nodo.tipo}")
        imprimir_arbol_simple(nodo.derecha)


# utils_tree.py (agregar al final del archivo)
import json

def serializar_arbol(nodo):
    if nodo is None:
        return None
    return {
        "tipo": nodo.tipo,
        "izquierda": serializar_arbol(nodo.izquierda),
        "derecha": serializar_arbol(nodo.derecha)
    }

def guardar_arbol_en_json(nodo_raiz, filename="arbol_usuarios.json"):
    """
    Guarda la estructura del árbol binario en un archivo JSON.
    """
    arbol_dict = serializar_arbol(nodo_raiz)
    with open(filename, "w") as f:
        json.dump(arbol_dict, f, indent=4)
