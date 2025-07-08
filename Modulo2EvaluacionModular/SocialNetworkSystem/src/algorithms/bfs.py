# Implementación del algoritmo de búsqueda en anchura (BFS) para sugerencias de amistad 
# en una red social, utilizando POO y principios SOLID.
# Algoritmo de sugerencia de amistad usando BFS
from models.user import Usuario
from models.network import RedSocial
from collections import deque



def sugerencias_amistad(red_social, nombre_usuario, max_sugerencias=3):
    """
    Algoritmo de sugerencia de amistad usando BFS.
    Recorre la red hasta 2 niveles de profundidad para sugerir amigos de amigos.
    """
    if nombre_usuario not in red_social.usuarios:
        raise ValueError(f"El usuario '{nombre_usuario}' no existe en la red.")

    usuario_origen = red_social.usuarios[nombre_usuario]

    '''
    Explorar amigos de un usuario (usuario_origen) en niveles (1er grado, 2do grado...) 
    y sugerir nuevos amigos (usuarios conectados a amigos pero no conectados directamente al usuario origen).
    '''
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
