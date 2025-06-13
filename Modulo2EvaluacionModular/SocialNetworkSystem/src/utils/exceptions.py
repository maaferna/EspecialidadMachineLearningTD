# Definición de excepciones personalizadas
# Definición de excepciones personalizadas para el sistema de red social

class UsuarioExistenteError(Exception):
    """Excepción cuando el usuario ya está registrado en la red social."""
    def __init__(self, nombre_usuario):
        self.nombre_usuario = nombre_usuario
        super().__init__(f"⚠️ El usuario '{nombre_usuario}' ya existe en la red.")

class UsuarioNoEncontradoError(Exception):
    """Excepción cuando no se encuentra un usuario en la red social."""
    def __init__(self, nombre_usuario):
        self.nombre_usuario = nombre_usuario
        super().__init__(f"❌ El usuario '{nombre_usuario}' no fue encontrado en la red.")
