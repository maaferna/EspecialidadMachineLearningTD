# Clase Usuario con POO y principios SOLID

class Usuario:
    """
    Clase que representa un usuario de la red social.
    """

    def __init__(self, nombre: str):
        self.nombre = nombre
        self.amigos = set()

    def agregar_amigo(self, otro_usuario: "Usuario") -> None:
        """
        Agrega una relación de amistad bidireccional.
        """
        if otro_usuario != self:
            self.amigos.add(otro_usuario)
            otro_usuario.amigos.add(self)

    def obtener_amigos(self) -> list:
        """
        Retorna una lista con los nombres de los amigos.
        """
        return [amigo.nombre for amigo in self.amigos]

    def __repr__(self):
        return f"Usuario({self.nombre})"
    
    def __str__(self):
        return f"Usuario: {self.nombre}, Amigos: {[amigo.nombre for amigo in self.amigos]}"