# Funciones auxiliares como mostrar formatear, validaciones, etc.


def formatear_amistades(nombre, amigos):
    """
    Retorna una cadena legible con la lista de amigos del usuario.
    """
    if not amigos:
        return f"👤 {nombre} no tiene amigos aún."
    amigos_str = ', '.join(amigos)
    return f"👤 {nombre} tiene como amigos: {amigos_str}"


def validar_conexion(usuario1, usuario2):
    """
    Verifica que no se intente conectar un usuario consigo mismo.
    """
    if usuario1 == usuario2:
        raise ValueError("⚠️ No se puede conectar un usuario consigo mismo.")


