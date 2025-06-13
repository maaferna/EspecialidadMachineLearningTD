# Implementación del algoritmo de búsqueda en anchura (BFS) para sugerencias de amistad 
# en una red social, utilizando POO y principios SOLID.
# Algoritmo de sugerencia de amistad usando BFS
from src.models.user import Usuario
from src.models.network import RedSocial
from collections import deque



def sugerencias_amistad(red_social, nombre_usuario, max_sugerencias=3):
    """
    Algoritmo de sugerencia de amistad usando BFS.
    Recorre la red hasta 2 niveles de profundidad para sugerir amigos de amigos.
    """
    if nombre_usuario not in red_social.usuarios:
        raise ValueError(f"El usuario '{nombre_usuario}' no existe en la red.")

    usuario_origen = red_social.usuarios[nombre_usuario]
    visitados = set()
    sugerencias = set()
    cola = deque([(usuario_origen, 0)])  # usuario y nivel

    while cola:
        actual, nivel = cola.popleft()

        if nivel > 2:
            break

        visitados.add(actual)

        for amigo in actual.amigos:
            if amigo not in visitados:
                cola.append((amigo, nivel + 1))
                if nivel == 1 and amigo not in usuario_origen.amigos:
                    sugerencias.add(amigo.nombre)

        if len(sugerencias) >= max_sugerencias:
            break

    return list(sugerencias)[:max_sugerencias]
