# 📱 Sistema Integrado de Gestión y Recomendación en una Red Social

Este proyecto representa una red social funcional desarrollada en Python. Implementa un sistema modular, escalable y optimizado para registrar usuarios, gestionar amistades y sugerir nuevas conexiones utilizando algoritmos eficientes y técnicas de optimización modernas.

---

## 🧩 Estructura del Proyecto

```

SocialNetworkSystem/
├── README.md
├── requirements.txt
├── environment.yml
├── data/
├── docs/
├── outputs/
│   ├── resultados\_tiempos.json
│   └── grafico\_comparacion\_tiempos.png
├── assets/
├── notebooks/
│   └── social\_network\_analysis.ipynb
├── tests/
├── src/
│   ├── **init**.py
│   ├── main.py                # Punto de entrada del sistema
│   ├── models/
│   │   ├── user.py            # Clase Usuario con diseño POO y principios SOLID
│   │   └── network.py         # Clase RedSocial
│   ├── utils/
│   │   ├── exceptions.py      # Excepciones personalizadas
│   │   ├── data\_generator.py  # Generación sintética de red social
│   │   └── helpers.py         # Funciones auxiliares
│   ├── algorithms/
│   │   └── bfs.py             # Algoritmo de búsqueda en anchura para recomendaciones
│   └── optimizations/
│       ├── timers.py          # Context manager para medir tiempo
│       └── optimized\_ops.py   # Implementaciones con Numba y NumPy

````

---

## ✅ Objetivos Cubiertos

* ✔ Gestión de usuarios y relaciones mediante diccionarios y grafos.
* ✔ Manejo robusto de excepciones (incluyendo personalizadas).
* ✔ Programación orientada a objetos con principios SOLID.
* ✔ Implementación de algoritmos eficientes (BFS).
* ✔ Comparación de rendimiento con `NumPy` y `Numba`.
* ✔ Visualización de resultados con `matplotlib`.
* ✔ Exportación de resultados en formato `.json`.


## 📌 Requerimientos Técnicos

### 1. Gestión de Datos

* Uso de estructuras: `dict`, `list`, `set`, `tuple`.
* Funciones implementadas:
  - `agregar_usuario(nombre)`
  - `conectar_usuarios(usuario1, usuario2)`
  - `obtener_red()`

### 2. Manejo de Excepciones

* Estructura `try` / `except` / `else` / `finally`.
* Excepciones personalizadas como:
  - `UsuarioExistenteError`
  - `UsuarioNoEncontradoError`

### 3. POO y Principios SOLID

* Clase `Usuario` encapsula los datos.
* Clase `RedSocial` gestiona usuarios y relaciones.
* Separación de responsabilidades clara:
  - Modelo de datos vs lógica de negocio vs optimización.

### 4. Algoritmos

* Búsqueda en anchura (`BFS`) para sugerencia de amigos.
* Análisis de complejidad con notación Big O.
* Módulo `bfs.py` implementa este comportamiento.

### 5. Optimización

* Comparación de tres enfoques:
  - Conjuntos (`set`)
  - Vectorización con `NumPy`
  - Compilación con `Numba` (JIT)
* Medición de tiempo usando `Timer` personalizado.

---

## 📊 Visualización y Resultados

### Archivos Generados

* 📄 [`outputs/resultados_tiempos.json`](outputs/resultados_tiempos.json): contiene los tiempos crudos de ejecución.
* 📈 [`outputs/grafico_comparacion_tiempos.png`](outputs/grafico_comparacion_tiempos.png): muestra la comparación gráfica entre `set` y `Numba`.

```json
[
  {
    "usuarios": 10000,
    "conjuntos": [0.00003, 0.00004, 0.00003],
    "numba": [0.9, 0.0078, 0.0077]
  },
  {
    "usuarios": 50000,
    "conjuntos": [...],
    "numba": [...]
  }
]
````

> 🔎 Observación: La primera ejecución con Numba es más lenta por la compilación JIT. Luego, mejora significativamente.

---

## 🧠 Observaciones

* La arquitectura del sistema sigue buenas prácticas de diseño profesional.
* El uso de Numba es más eficiente para grandes volúmenes de datos.
* El proyecto puede ser ampliado con:

  * Base de datos para persistencia
  * Sistema de login/autenticación
  * API REST con Flask o FastAPI

---

## 💡 Ejecución

### Desde línea de comandos

```bash
python src/main.py
```

### Desde Jupyter Notebook

```bash
jupyter notebook notebooks/social_network_analysis.ipynb
```

---

## 🛠 Requerimientos de Entorno

### Paquetes Principales

```txt
numpy
numba
matplotlib
faker
pandas
```

### Instalación con Conda

```bash
conda env create -f environment.yml
conda activate especialidadmachinelearning
```

---

## 👨‍🏫 Evaluación

Este proyecto fue desarrollado como parte del módulo de Programación Avanzada del curso de **Especialización en Machine Learning**, integrando:

* Estructuras avanzadas
* POO y SOLID
* Algoritmos eficientes
* Optimización y visualización
* Documentación profesional

---

### 📈 Análisis de Resultados

A partir de los datos almacenados en [`outputs/resultados_tiempos.json`](outputs/resultados_tiempos.json) y visualizados en el gráfico [`grafico_comparacion_tiempos.png`](outputs/grafico_comparacion_tiempos.png), se obtuvieron las siguientes conclusiones:

#### ⏱️ Rendimiento

| Número de Usuarios | Tiempo (Set - promedio) | Tiempo (Numba - promedio) |
| ------------------ | ----------------------- | ------------------------- |
| 10,000             | \~0.00003 segundos      | \~0.0077 segundos         |
| 50,000             | \~0.00003 segundos      | \~0.0054 segundos         |
| 100,000            | \~0.00002 segundos      | \~0.0051 segundos         |

#### 🧠 Observaciones

* El uso de `set` (intersección de conjuntos) es **extremadamente rápido** y constante, incluso para 100.000 usuarios, debido a su eficiencia en operaciones de pertenencia (O(1)).
* El uso de `Numba` presenta un **retardo inicial** debido a la **compilación Just-In-Time (JIT)**, pero luego logra tiempos comparables y adecuados.
* En algunos escenarios, `Numba` no supera a `set`, lo cual **es esperable**, ya que `set` es una de las operaciones más optimizadas de Python para comparación entre colecciones pequeñas o medianas.
* A mayor escala o con estructuras de datos más densas y numéricas (e.g., matrices de adyacencia), **`Numba` podría ofrecer mejores resultados relativos**.

#### 📌 Conclusión

* Para redes sociales pequeñas o medianas, el uso de `set` es suficiente y altamente eficiente.
* Para escalabilidad o integración con cálculos numéricos más pesados, `Numba` ofrece una base sólida para futuras optimizaciones.

