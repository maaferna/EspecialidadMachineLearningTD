from src.utils.exceptions import UsuarioExistenteError, UsuarioNoEncontradoError
from src.utils.data_generator import generar_red_social
from src.models.network import RedSocial
from src.optimizations.optimized_ops import (
    calcular_amigos_en_comun,
    calcular_amigos_en_comun_numba,
    convertir_amigos_a_numpy
)
from src.optimizations.timers import Timer

import numpy as np


# Crear instancia de RedSocial
red_social = RedSocial()

# Crear usuarios y conexiones usando datos generados
red_dict = generar_red_social(n_usuarios=10000, max_amigos=100)
for usuario, amigos in red_dict.items():
    try:
        red_social.agregar_usuario(usuario)
    except UsuarioExistenteError as e:
        print(e)

for usuario, amigos in red_dict.items():
    for amigo in amigos:
        try:
            red_social.conectar_usuarios(usuario, amigo)
        except UsuarioNoEncontradoError:
            # Esto puede ocurrir si algún usuario no fue agregado por error de Faker
            continue

# Obtener estructura final como diccionario plano
red = red_social.obtener_red()

# Seleccionar dos usuarios para prueba
usuarios = list(red.keys())
user_a, user_b = usuarios[0], usuarios[1]

amigos_a = red[user_a]
amigos_b = red[user_b]

# Cálculo con conjuntos
with Timer("Amigos en común (conjuntos)"):
    comunes_set = calcular_amigos_en_comun(amigos_a, amigos_b)

# Cálculo con Numba
amigos_np = convertir_amigos_a_numpy({user_a: amigos_a, user_b: amigos_b})
with Timer("Amigos en común (Numba)"):
    comunes_numba = calcular_amigos_en_comun_numba(amigos_np[user_a], amigos_np[user_b])
