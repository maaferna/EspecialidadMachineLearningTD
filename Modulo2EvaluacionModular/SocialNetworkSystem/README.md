# 📱 Sistema Integrado de Gestión y Recomendación en una Red Social
# Sistema Integrado de Gestión y Recomendación en una Red Social

Este proyecto representa una red social funcional desarrollada en Python. Implementa un sistema modular, escalable y optimizado para registrar usuarios, gestionar amistades y sugerir nuevas conexiones utilizando algoritmos eficientes.

## 🧩 Estructura del Proyecto

```
SocialNetworkSystem/
├── README.md
├── requirements.txt
├── data/
├── docs/
├── outputs/
├── assets/
├── notebooks/
├── tests/
├── src/
│   ├── __init__.py
│   ├── main.py                # Punto de entrada del sistema
│   ├── models/
│   │   └── user.py            # Clase Usuario con diseño POO y principios SOLID
│   ├── utils/
│   │   ├── exceptions.py      # Excepciones personalizadas
│   │   └── helpers.py         # Funciones auxiliares como mostrar menú, validaciones, etc.
│   ├── algorithms/
│   │   └── bfs.py             # Algoritmo de búsqueda en anchura para recomendaciones
│   └── optimizations/
│       ├── timers.py          # Context manager para medir tiempo
│       └── optimized_ops.py   # Implementaciones con Numba y NumPy
```

## ✅ Objetivos Cubiertos

* ✔ Gestión de usuarios y relaciones mediante diccionarios y grafos.
* ✔ Manejo robusto de excepciones (incluyendo personalizadas).
* ✔ Programación orientada a objetos con principios SOLID.
* ✔ Implementación de algoritmos (búsqueda en anchura - BFS).
* ✔ Optimización del rendimiento con `NumPy` y `Numba`.
* ✔ Visualización de resultados y documentación completa.

## 📌 Requerimientos Técnicos

### 1. Gestión de Datos

* `agregar_usuario(nombre)`
* `agregar_amigo(usuario_a, usuario_b)`

### 2. Manejo de Excepciones

* Estructuras `try`, `except`, `else`, `finally`.
* Excepciones personalizadas en `exceptions.py`.

### 3. POO con Principios SOLID

* Clase `Usuario` (encapsulamiento, SRP)
* Separación de lógica de recomendación, visualización y modelo de datos.

### 4. Algoritmos

* `BFS` para sugerencia de amistad.
* Complejidad analizada con Big O.
* Visualización con `matplotlib` de tiempos.

### 5. Optimización

* Comparaciones de rendimiento entre implementación básica, NumPy y Numba.
* Uso del context manager `Timer`.

## 📊 Visualización y Resultados

* Los resultados de los tiempos y recomendaciones se almacenan en JSON y se grafican automáticamente.

## 💡 Ejecución

Desde la raíz del proyecto:

```bash
python src/main.py
```

## 🛠 Requerimientos de Entorno

```txt
numpy
numba
matplotlib
networkx
pandas
```

Instalación:

```bash
conda create -n socialnet python=3.8
conda activate socialnet
```

---

## 🧠 Observaciones

* Este sistema sigue buenas prácticas de diseño.
* Su arquitectura modular permite escalabilidad y pruebas unitarias.
* Se recomienda extender con base de datos y autenticación para producción.
