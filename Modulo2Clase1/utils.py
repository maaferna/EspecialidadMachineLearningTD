# utils.py

import json
import os

# -----------------------------
# 📁 Archivos utilizados para persistencia
# -----------------------------
USUARIOS_FILE = "usuarios.json"
RED_FILE = "red_conexiones.json"
ACTIVIDADES_FILE = "historial_actividades.json"
SOLICITUDES_FILE = "solicitudes_amistad.json"

# -----------------------------
# 🔄 Cargar usuarios y red de conexiones
# -----------------------------
def cargar_datos():
    """
    Carga los datos de usuarios y red de conexiones desde archivos JSON.
    
    Returns:
        tuple: Diccionarios de usuarios y red_conexiones.
    """
    if os.path.exists(USUARIOS_FILE):
        with open(USUARIOS_FILE, "r") as f:
            usuarios = json.load(f)
    else:
        usuarios = {}

    if os.path.exists(RED_FILE):
        with open(RED_FILE, "r") as f:
            red_conexiones = json.load(f)
    else:
        red_conexiones = {}

    return usuarios, red_conexiones

# -----------------------------
# 💾 Guardar usuarios y red de conexiones
# -----------------------------
def guardar_datos(usuarios, red_conexiones):
    """
    Guarda los datos de usuarios y conexiones en archivos JSON.
    
    Args:
        usuarios (dict): Información de los usuarios.
        red_conexiones (dict): Grafo de conexiones.
    """
    with open(USUARIOS_FILE, "w") as f:
        json.dump(usuarios, f, indent=4)
    with open(RED_FILE, "w") as f:
        json.dump(red_conexiones, f, indent=4)

# -----------------------------
# 🔁 Sincronizar amigos en usuarios con red_conexiones
# -----------------------------
def sincronizar_amigos(usuarios, red_conexiones):
    """
    Actualiza la clave 'amigos' de cada usuario en el diccionario 'usuarios' 
    para reflejar las conexiones actuales en 'red_conexiones'.
    
    Args:
        usuarios (dict): Diccionario de usuarios.
        red_conexiones (dict): Grafo de conexiones.
    """
    for usuario in usuarios:
        usuarios[usuario]['amigos'] = red_conexiones.get(usuario, [])
    guardar_datos(usuarios, red_conexiones)

# -----------------------------
# ✅ Función para agregar un usuario
# -----------------------------
def agregar_usuario(username, nombre, edad, usuarios, red_conexiones):
    """
    Agrega un nuevo usuario al sistema, actualizando la base de datos de usuarios
    y la red de conexiones (Diccionarios).

    Args:
        username (str): Identificador único del usuario (nombre de usuario).
        nombre (str): Nombre completo del usuario.
        edad (int): Edad del usuario.
    """
    if username not in usuarios:
        usuarios[username] = {
            'nombre': nombre,
            'edad': edad,
            'amigos': []
        }
        red_conexiones[username] = []
        guardar_datos(usuarios, red_conexiones)
    else:
        print(f"⚠️ El usuario '{username}' ya existe.")

# -----------------------------
# ✅ Función para conectar dos usuarios (grafo no dirigido)
# -----------------------------
def agregar_amigo(user1, user2, usuarios, red_conexiones):
    """
    Conecta dos usuarios en la red social simulada, actualizando el grafo de conexiones.

    Args:
        user1 (str): Nombre de usuario del primer usuario.
        user2 (str): Nombre de usuario del segundo usuario.
    """
    if user1 in red_conexiones and user2 in red_conexiones:
        if user2 not in red_conexiones[user1]:
            red_conexiones[user1].append(user2)
            red_conexiones[user2].append(user1)
            sincronizar_amigos(usuarios, red_conexiones)
        else:
            print(f"⚠️ '{user1}' y '{user2}' ya están conectados.")
    else:
        print("❌ Uno o ambos usuarios no existen.")

# -----------------------------
# 🔄 Cargar historial y solicitudes (pilas y colas)
# -----------------------------
def cargar_pilas_colas():
    """
    Carga el historial de actividades y la cola de solicitudes desde archivos JSON.
    
    Returns:
        tuple: Lista de actividades y lista de solicitudes.
    """
    if os.path.exists(ACTIVIDADES_FILE):
        with open(ACTIVIDADES_FILE, "r") as f:
            historial = json.load(f)
    else:
        historial = []

    if os.path.exists(SOLICITUDES_FILE):
        with open(SOLICITUDES_FILE, "r") as f:
            solicitudes = json.load(f)
    else:
        solicitudes = []

    return historial, solicitudes

# -----------------------------
# 💾 Guardar historial y solicitudes
# -----------------------------
def guardar_pilas_colas(historial, solicitudes):
    """
    Guarda el historial de actividades y solicitudes de amistad.
    
    Args:
        historial (list): Pila de actividades.
        solicitudes (list): Cola de solicitudes.
    """
    with open(ACTIVIDADES_FILE, "w") as f:
        json.dump(historial, f, indent=4)
    with open(SOLICITUDES_FILE, "w") as f:
        json.dump(solicitudes, f, indent=4)

# -----------------------------
# 📦 Funciones para la pila de actividades
# -----------------------------
def push_actividad(usuario, descripcion, historial, solicitudes):
    """
    Registra una actividad realizada por un usuario.

    Args:
        usuario (str): Usuario que realizó la actividad.
        descripcion (str): Descripción de la actividad.
    """
    actividad = {"usuario": usuario, "descripcion": descripcion}
    historial.append(actividad)
    guardar_pilas_colas(historial, solicitudes)

def pop_actividad(historial, solicitudes):
    """
    Elimina la última actividad registrada.

    Returns:
        dict|str: Actividad eliminada o mensaje si está vacía.
    """
    if historial:
        actividad = historial.pop()
        guardar_pilas_colas(historial, solicitudes)
        return actividad
    return "Sin actividades registradas."

# -----------------------------
# 🧾 Funciones para la cola de solicitudes de amistad
# -----------------------------
def enqueue_solicitud(remitente, destinatario, historial, solicitudes):
    """
    Agrega una solicitud de amistad entre dos usuarios.

    Args:
        remitente (str): Usuario que envía la solicitud.
        destinatario (str): Usuario que la recibe.
    """
    solicitud = {"de": remitente, "para": destinatario}
    solicitudes.append(solicitud)
    guardar_pilas_colas(historial, solicitudes)

def dequeue_solicitud(historial, solicitudes):
    """
    Procesa la solicitud más antigua de la cola.

    Returns:
        dict|str: Solicitud procesada o mensaje si no hay solicitudes.
    """
    if solicitudes:
        solicitud = solicitudes.pop(0)
        guardar_pilas_colas(historial, solicitudes)
        return solicitud
    return "No hay solicitudes pendientes."

