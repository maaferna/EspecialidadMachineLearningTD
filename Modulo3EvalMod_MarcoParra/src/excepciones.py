class MatrizSingularError(Exception):
    """
    Excepción personalizada para errores al invertir la matriz XᵀX en regresión lineal.
    """
    def __init__(self, mensaje="La matriz XᵀX es singular y no se puede invertir."):
        self.mensaje = mensaje
        super().__init__(self.mensaje)


class ParametrosNoConvergentesError(Exception):
    """
    Excepción personalizada para errores al no converger los parámetros durante el entrenamiento.
    """
    def __init__(self, mensaje="Los parámetros no convergen dentro del número máximo de iteraciones."):
        self.mensaje = mensaje
        super().__init__(self.mensaje)

class DatosInsuficientesError(Exception):
    """
    Excepción personalizada para errores al intentar entrenar con datos insuficientes.
    """
    def __init__(self, mensaje="No hay suficientes datos para entrenar el modelo."):
        self.mensaje = mensaje
        super().__init__(self.mensaje)