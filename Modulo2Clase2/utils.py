import os
import json

# 1. 📂 Función que procesa un archivo línea por línea
def procesar_archivo(ruta_archivo):
    """
    Abre un archivo de texto, intenta convertir cada línea a entero
    y calcula 100 dividido por ese valor. Maneja múltiples excepciones
    y lanza una excepción personalizada si el número es negativo.
    """
    try:
        with open(ruta_archivo, 'r') as archivo:
            print(f"📄 Procesando archivo: {ruta_archivo}")
            for i, linea in enumerate(archivo, start=1):
                try:
                    valor = int(linea.strip())
                    if valor < 0:
                        raise ValorNegativoError(f"Línea {i}: Se encontró un valor negativo ({valor})")
                    resultado = 100 / valor
                    print(f"Línea {i}: 100 / {valor} = {resultado}")
                except ValueError:
                    print(f"❌ Línea {i}: No es un número válido -> '{linea.strip()}'")
                except ZeroDivisionError:
                    print(f"❌ Línea {i}: División entre cero no permitida")
                except ValorNegativoError as e:
                    print(f"⚠️ {e}")
                else:
                    print(f"✅ Línea {i} procesada correctamente")
                finally:
                    print(f"🧹 Línea {i}: Limpieza final\n")
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {ruta_archivo}")
    else:
        print("✅ Archivo procesado correctamente")
    finally:
        print("🔚 Finalizando ejecución de procesar_archivo()\n")

# 3. ❗ Excepción personalizada
class ValorNegativoError(Exception):
    """Excepción lanzada cuando se encuentra un valor negativo no permitido."""
    pass


