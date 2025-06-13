from faker import Faker
import random


def generar_red_social(n_usuarios=1000, max_amigos=50, seed=42):
    """
    Genera una red social sintética con usuarios y conexiones.

    Args:
        n_usuarios (int): Número de usuarios.
        max_amigos (int): Máximo de amigos por usuario.
        seed (int): Semilla para reproducibilidad.

    Returns:
        dict[str, list[str]]: Diccionario {usuario: [amigos]}
    """
    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)

    usuarios = [fake.unique.user_name() for _ in range(n_usuarios)]
    red = {}

    for usuario in usuarios:
        n_amigos = random.randint(1, max_amigos)
        posibles_amigos = list(set(usuarios) - {usuario})
        amigos = random.sample(posibles_amigos, min(n_amigos, len(posibles_amigos)))
        red[usuario] = amigos

    return red
