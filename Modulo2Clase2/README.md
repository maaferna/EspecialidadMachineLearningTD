# Proyecto: Manejo de Excepciones en Python

Este proyecto tiene como objetivo demostrar el dominio de conceptos teóricos y prácticos sobre el manejo de excepciones en Python. Se implementa una función de procesamiento de archivos que maneja distintos tipos de errores, incluyendo excepciones personalizadas.

## Estructura del Proyecto

```
Modulo2Clase2/
│
├── main.py                # Script principal para ejecutar la función de 
├── utils.py               # Contiene la lógica de procesamiento y manejo de 
├── archivo_prueba.txt     # Archivo de texto con datos de prueba
└── README.md              # Documentación del proyecto
```

## Contenido del Proyecto

### 1. `main.py`

Contiene el código que:

* Crea un archivo de prueba (`archivo_prueba.txt`) con datos variados.
* Invoca la función `procesar_archivo()` importada desde `utils.py`.

### 2. `utils.py`

Incluye:

* La función `procesar_archivo(ruta_archivo)` que:

  * Lee línea por línea un archivo.
  * Intenta convertir cada línea a un entero.
  * Realiza la operación `100 / numero`.
* Manejo de excepciones:

  * `FileNotFoundError` si el archivo no existe.
  * `ValueError` si el contenido no puede convertirse a entero.
  * `ZeroDivisionError` para divisiones por cero.
  * Clausulas `else` y `finally` para mostrar el flujo correcto.
* Excepción personalizada:

  * `ValorNegativoError`: se lanza cuando el número es negativo, heredando de `Exception`.

### 3. `archivo_prueba.txt`

Archivo de texto de prueba con los siguientes datos:

```
25
0
abc
-10
5
```

## Ejemplo de Ejecución

```
📄 Procesando archivo: archivo_prueba.txt
Línea 1: 100 / 25 = 4.0
✅ Línea 1 procesada correctamente
🧹 Línea 1: Limpieza final
...
✅ Archivo procesado correctamente
🔚 Finalizando ejecución de procesar_archivo()
```

## Requisitos

* Python 3.x

## Ejecución

Desde la carpeta del proyecto:

```bash
python main.py
```

## Autores

* Proyecto desarrollado como parte del Módulo 2 - Clase 2: Especialidad en Machine Learning (Bootcamp)

---

Este proyecto demuestra el uso práctico de estructuras de control para el manejo robusto de errores, asegurando estabilidad y claridad en flujos de procesamiento de datos.
