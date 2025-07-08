from models.user import Usuario
from utils.exceptions import UsuarioExistenteError, UsuarioNoEncontradoError


class RedSocial:
    """
    Clase que gestiona la red social.
    """

    def __init__(self):
        self.usuarios = {}

    def agregar_usuario(self, nombre: str):
        if nombre in self.usuarios:
            raise UsuarioExistenteError(f"⚠️ El usuario '{nombre}' ya existe.")
        self.usuarios[nombre] = Usuario(nombre)

    def conectar_usuarios(self, nombre1: str, nombre2: str):
        if nombre1 not in self.usuarios or nombre2 not in self.usuarios:
            raise UsuarioNoEncontradoError("❌ Uno o ambos usuarios no existen.")
        self.usuarios[nombre1].agregar_amigo(self.usuarios[nombre2])

    def obtener_red(self):
        return {
            nombre: usuario.obtener_amigos()
            for nombre, usuario in self.usuarios.items()
        }
