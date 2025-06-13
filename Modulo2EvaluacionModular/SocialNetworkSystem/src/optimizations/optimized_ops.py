# Implementaciones con NumPy y Numba

import numpy as np
from numba import jit


def calcular_amigos_en_comun(user_friends, target_friends):
    """
    Calcula amigos en común usando operaciones vectorizadas.
    
    Args:
        user_friends (list[int]): IDs de amigos del usuario.
        target_friends (list[int]): IDs de amigos del usuario objetivo.

    Returns:
        int: Número de amigos en común.
    """
    return len(set(user_friends).intersection(set(target_friends)))


@jit(nopython=True)
def calcular_amigos_en_comun_numba(arr1, arr2):
    """
    Calcula amigos en común usando Numba para acelerar en listas grandes.
    Ideal cuando los datos provienen de grafos muy densos.
    
    Args:
        arr1, arr2: arrays tipo NumPy de enteros.

    Returns:
        int: Número de coincidencias.
    """
    count = 0
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            if arr1[i] == arr2[j]:
                count += 1
    return count


def convertir_amigos_a_numpy(friend_dict):
    """
    Convierte un diccionario de amigos por usuario en arrays NumPy para procesamientos masivos.

    Args:
        friend_dict (dict[str, list[int]]): Diccionario {usuario: [amigos]}.

    Returns:
        dict[str, np.ndarray]: Diccionario {usuario: np.array(amigos)}
    """
    return {user: np.array(friends, dtype=np.int32) for user, friends in friend_dict.items()}
