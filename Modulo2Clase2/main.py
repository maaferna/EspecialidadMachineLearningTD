from utils import procesar_archivo, ValorNegativoError

# 2. 🛠️ Importar la función procesar_archivo y la excepción personalizada
# Crear archivo de prueba
archivo_test = "archivo_prueba.txt"
with open(archivo_test, "w") as f:
    f.write("25\n")
    f.write("0\n")
    f.write("abc\n")
    f.write("-10\n")
    f.write("5\n")

# Ejecutar función de prueba
procesar_archivo(archivo_test)

archivo_no_existe = "archivo_prueba_no_existe.txt"
with open(archivo_test, "w") as f:
    f.write("25\n")
    f.write("0\n")
    f.write("abc\n")
    f.write("-10\n")
    f.write("5\n")

procesar_archivo(archivo_no_existe)