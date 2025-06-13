import json

class Vehiculo:
    """
    Clase base que representa un vehículo genérico.

    Esta clase define atributos comunes para cualquier tipo de vehículo y 
    métodos básicos que pueden ser sobrescritos por clases hijas.

    Atributos:
        __marca (str): Marca del vehículo.
        __modelo (str): Modelo del vehículo.
        __año (int): Año de fabricación del vehículo.
    """

    def __init__(self, marca, modelo, año):
        """
        Inicializa un objeto Vehiculo con los atributos básicos.

        Args:
            marca (str): Marca del vehículo (por ejemplo, 'Toyota').
            modelo (str): Modelo del vehículo (por ejemplo, 'Corolla').
            año (int): Año de fabricación del vehículo (por ejemplo, 2020).
        """
        self.__marca = marca
        self.__modelo = modelo
        self.__año = año

    def acelerar(self):
        """Imprime un mensaje genérico indicando que el vehículo está acelerando."""
        print("El vehículo está acelerando de forma genérica.")

    def obtener_info(self):
        """
        Devuelve una cadena con la información básica del vehículo.

        Returns:
            str: Información formateada, por ejemplo: 'Toyota Corolla (2020)'.
        """
        return f"{self.__marca} {self.__modelo} ({self.__año})"
    
    def to_dict(self):
        """
        Devuelve los atributos del vehículo en formato diccionario.

        Returns:
            dict: Información del vehículo como diccionario.
        """
        return {
            "marca": self.__marca,
            "modelo": self.__modelo,
            "año": self.__año
        }


class Auto(Vehiculo):
    """
    Clase derivada que representa un automóvil.
    """

    def acelerar(self):
        """Mensaje específico para aceleración de autos."""
        print("🚗 El auto acelera suavemente en carretera.")

    def abrir_maletero(self):
        """Simula abrir el maletero del auto."""
        print("El maletero del auto ha sido abierto.")


class Moto(Vehiculo):
    """
    Clase derivada que representa una motocicleta.
    """

    def acelerar(self):
        """Mensaje específico para aceleración de motocicletas."""
        print("🏍️ La moto acelera rápidamente en la autopista.")

    def hacer_caballito(self):
        """Simula realizar un caballito con la moto."""
        print("La moto está haciendo un caballito. ¡Cuidado!")


class VehiculoPrinter:
    """
    Clase dedicada a mostrar información de vehículos por consola.
    (Responsabilidad única: visualización humana).
    """

    @staticmethod
    def mostrar_info(vehiculo):
        print(f"📋 Información del vehículo: {vehiculo.obtener_info()}")


class VehiculoJSONPrinter:
    """
    Clase dedicada a exportar información de vehículos en formato JSON.
    (Responsabilidad única: persistencia o interoperabilidad).
    """

    @staticmethod
    def exportar_a_json(vehiculo):
        """
        Convierte los datos del vehículo a una cadena JSON.

        Args:
            vehiculo (Vehiculo): Objeto del que se desea obtener información.

        Returns:
            str: Cadena JSON.
        """
        return json.dumps(vehiculo.to_dict(), indent=4, ensure_ascii=False)

    @staticmethod
    def guardar_en_archivo(vehiculo, ruta):
        """
        Guarda los datos del vehículo como archivo JSON.

        Args:
            vehiculo (Vehiculo): Objeto a guardar.
            ruta (str): Ruta del archivo de destino.
        """
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(vehiculo.to_dict(), f, indent=4, ensure_ascii=False)

