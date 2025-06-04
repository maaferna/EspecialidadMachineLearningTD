import pprint
from utils import *
from utils_tree import NodoBasico, guardar_arbol_en_json, imprimir_arbol_simple

# -----------------------------
# 🔄 Inicialización de estructuras con persistencia
# -----------------------------
try:
    usuarios, red_conexiones = cargar_datos()
except FileNotFoundError:
    usuarios, red_conexiones = {}, {}
    print("Archivos no encontrados, se inicializan estructuras vacías.")

# -----------------------------
# 🧪 Datos de prueba: Usuarios y conexiones
# -----------------------------
agregar_usuario("juan23", "Juan Pérez", 30, usuarios, red_conexiones)
agregar_usuario("lore89", "Lorena Díaz", 27, usuarios, red_conexiones)
agregar_usuario("mike91", "Miguel Torres", 32, usuarios, red_conexiones)

agregar_amigo("juan23", "lore89", usuarios, red_conexiones)
agregar_amigo("juan23", "mike91", usuarios, red_conexiones)

# -----------------------------
# 🖨️ Visualización de resultados
# -----------------------------
print("\n📋 Usuarios registrados:")
pprint.pprint(usuarios)

print("\n🔗 Red de conexiones:")
pprint.pprint(red_conexiones)

print("*"*50)

# -----------------------------
# ✅ Listas y Tuplas: slicing e indexación
# -----------------------------
lista_usuarios = list(usuarios.keys())
primeros_dos_usuarios = lista_usuarios[:2]
primer_usuario = lista_usuarios[0]
ultimo_usuario = lista_usuarios[-1]

roles = ("admin", "moderador", "usuario")
rol_predeterminado = roles[2]
try:
    roles[0] = "superadmin"
except TypeError as e:
    print(f"Error al intentar modificar la tupla: {e}")

print(f"\nRol predeterminado: {rol_predeterminado}")
print({
    "lista_usuarios": lista_usuarios,
    "primeros_dos_usuarios": primeros_dos_usuarios,
    "primer_usuario": primer_usuario,
    "ultimo_usuario": ultimo_usuario,
    "roles": roles,
    "rol_predeterminado": rol_predeterminado
})
print(f"Primer rol: {roles[0]}, Último rol: {roles[-1]}")

print("*"*50)

# -----------------------------
# 📦 PILAS y 🧾 COLAS: con persistencia
# -----------------------------
historial_actividades, solicitudes_amistad = cargar_pilas_colas()

# Actividades (stack)
push_actividad("juan23", "Publicó una foto", historial_actividades, solicitudes_amistad)
push_actividad("juan23", "Comentó en una publicación", historial_actividades, solicitudes_amistad)
push_actividad("juan23", "Actualizó su perfil", historial_actividades, solicitudes_amistad)

ultima_actividad = pop_actividad(historial_actividades, solicitudes_amistad)

# Solicitudes (queue)
enqueue_solicitud("lore89", "juan23", historial_actividades, solicitudes_amistad)
enqueue_solicitud("mike91", "juan23", historial_actividades, solicitudes_amistad)

solicitud_procesada = dequeue_solicitud(historial_actividades, solicitudes_amistad)

print("\nHistorial de actividades:", historial_actividades)
print("\nÚltima actividad removida:", ultima_actividad)
print("\nSolicitudes de amistad:", solicitudes_amistad)
print("\nSolicitud procesada:", solicitud_procesada)

# -----------------------------
# 🧪 Prueba de Árbol Binario de Usuarios
# -----------------------------
# Crear nodos manualmente (sin lógica de inserción avanzada)
raiz = NodoBasico("Administrador")
raiz.izquierda = NodoBasico("Moderador")
raiz.derecha = NodoBasico("Invitado")
raiz.izquierda.izquierda = NodoBasico("Usuario registrado")

# -----------------------------
# 🧪 Mostrar jerarquía simple
# -----------------------------
print("🌳 Jerarquía de tipos de usuario (versión simple):")
imprimir_arbol_simple(raiz)

# -----------------------------
# 💾 Guardar árbol en JSON
# -----------------------------
guardar_arbol_en_json(raiz)
print("✅ Árbol guardado como 'arbol_usuarios.json'")
