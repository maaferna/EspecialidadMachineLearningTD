# 📊 Comparación de Métodos de Optimización en Regresión Lineal

Este proyecto implementa y compara distintos métodos de optimización para minimizar una función de costo en un problema de **regresión lineal**.

Se evalúan dos técnicas clásicas:
- **Gradient Descent (GD)**
- **Stochastic Gradient Descent (SGD)**

Y se analizan en términos de:
- Convergencia del error (MSE)
- Evolución de parámetros (pendiente `w` y sesgo `b`)
- Estabilidad del entrenamiento

---

## 🎯 Objetivo

Aplicar y comparar algoritmos de optimización sobre una función de costo en regresión lineal:
- Generar datos sintéticos con ruido
- Definir y derivar la función de costo (MSE)
- Calcular gradientes
- Aplicar GD y SGD para minimizar el error
- Visualizar y analizar los resultados

---

## 🧠 Stack Tecnológico

- **Python 3.8**
- **NumPy**: manipulación matemática
- **Matplotlib**: visualización
- **Conda** (opcional): manejo de entorno virtual

---

## 📂 Estructura del Proyecto

```

Modulo3EvalMod\_MarcoParra/
├── environment.yml            # Dependencias Conda (auto-generado)
├── init\_project.sh           # Script para crear estructura del proyecto
├── README.md                 # Este archivo
├── notebooks/                # Notebooks generados automáticamente
│   └── comparacion\_regresion.ipynb
├── outputs/                  # Resultados (gráficos)
│   └── comparacion\_gd\_sgd.png
├── scripts/                  # Punto de entrada principal
│   └── main.py
└── src/                      # Código modular del proyecto
├── datos.py              # Generación de datos sintéticos
├── optimizadores.py      # Implementación de GD y SGD
├── utils.py              # MSE y cálculo de gradientes
└── visualizador.py       # Funciones gráficas

````

---

## ⚙️ Instalación

### ✅ Opción 1: Usando Conda (recomendado)

```bash
conda env create -f environment.yml
conda activate especialidadmachinelearning
````

> El archivo `environment.yml` se genera automáticamente por el script `init_project.sh`.

### ✅ Opción 2: Sin Conda (instalar manualmente)

```bash
pip install numpy matplotlib
```

---

## 🚀 Uso

### ⚡ Crear la estructura del proyecto

```bash
chmod +x init_project.sh
./init_project.sh
```

Esto:

* Genera carpetas (`src`, `scripts`, `outputs`, `notebooks`)
* Crea un `main.py` con lógica inicial
* Crea `environment.yml`
* Crea módulos `.py` básicos para iniciar

### ▶️ Ejecutar el análisis principal

```bash
python scripts/main.py
```

Esto ejecutará el entrenamiento con GD y SGD y guardará el gráfico comparativo en `outputs/comparacion_gd_sgd.png`.

---

## 📓 Jupyter Notebook

Un notebook con la lógica del proyecto puede ser generado automáticamente en `notebooks/` si se implementa `crear_notebook.py`.

---

## 🧠 Reflexión

Este proyecto muestra cómo la elección del método de optimización y la tasa de aprendizaje afectan:

* la estabilidad del entrenamiento,
* la velocidad de convergencia,
* y la precisión de los parámetros finales.

Esto es fundamental en Machine Learning, donde entrenar modelos eficientes depende de una buena estrategia de optimización.

---

# 📈 Comparación de Métodos de Optimización para Regresión Lineal

Este proyecto implementa, compara y analiza diferentes enfoques para resolver un problema de regresión lineal simple, utilizando tanto métodos analíticos como iterativos (descenso de gradiente y su variante estocástica).


---

## 📊 Metodología

1. **Generación de Datos**

   * Se generan 100 puntos según:

     $$
     y = 4 + 3x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
     $$

2. **Resolución del Modelo**

   * **Cálculo Cerrado**:

     $$
     \beta = (X^T X)^{-1} X^T y
     $$
   * Validación mediante manejo de excepciones si la matriz es singular.

3. **Optimización Iterativa**:

   * **GD**: utiliza todo el conjunto de datos por iteración.
   * **SGD**: actualiza con una muestra por iteración.

4. **Visualización**:

   * Evolución del costo (MSE) a lo largo de las iteraciones.
   * Trayectoria de los parámetros `w` y `b`.

---

## 📉 Resultados Obtenidos

### ⚡ Cálculo Cerrado

* Parámetros analíticos:

  $$
  w = 2.9540,\quad b = 4.2151
  $$

### 🔁 Gradient Descent (GD)

* Parámetros finales:

  $$
  w = 3.1397,\quad b = 3.0072
  $$

### 🔁 Stochastic Gradient Descent (SGD)

* Parámetros finales:

  $$
  w = 3.0165,\quad b = 3.9912
  $$

---

## 📷 Visualización de Resultados

📍 Output guardado en [`outputs/comparacion_gd_sgd.png`](outputs/comparacion_gd_sgd.png)

![Comparación GD vs SGD](outputs/comparacion_gd_sgd.png)

* La curva de costo de **GD** es más suave, pero más lenta en converger.
* **SGD** desciende rápidamente al inicio, pero con mayor variabilidad.
* Ambos métodos se aproximan al modelo analítico, validando su correcto funcionamiento.

---

## 📌 Reflexión Final

* **GD** y **SGD** convergen hacia la solución óptima, aunque con diferentes trayectorias y eficiencia.
* **SGD** puede ser más adecuado en grandes volúmenes de datos o en entornos en línea.
* El **cálculo cerrado** es ideal para problemas pequeños y bien condicionados.
* La **tasa de aprendizaje (learning rate)** juega un rol clave en la convergencia y debe ajustarse cuidadosamente.

---

## 🛡️ Manejo de Errores

Se definen excepciones personalizadas para:

* `MatrizSingularError`: si la matriz $X^T X$ no puede invertirse.
* `ParametrosNoConvergentesError`: si el costo no mejora significativamente.
* `DatosInsuficientesError`: si los datos no son suficientes para entrenamiento.

---

## ✅ Validación del Proyecto con Tests Automatizados

Este proyecto incluye un conjunto de pruebas para validar el correcto funcionamiento de las excepciones personalizadas definidas en `src/excepciones.py`. Estas pruebas aseguran que el sistema responde adecuadamente ante situaciones específicas como:

* **Singularidad de la matriz** en la regresión lineal cerrada.
* **Falta de convergencia** en algoritmos iterativos.
* **Datos insuficientes** para entrenar un modelo.

### 📁 Ubicación de los tests

Los tests están ubicados en el directorio:

```
tests/test_excepciones.py
```

### ▶️ Ejecución de los tests

Desde la raíz del proyecto, ejecuta el siguiente comando:

```bash
pytest tests/test_excepciones.py -v
```

### ✅ Resultado esperado

Una ejecución exitosa mostrará una salida como la siguiente:

```
==================================================== test session starts =====================================================
platform linux -- Python 3.11.6, pytest-7.1.3, pluggy-1.0.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/mparraf/myprojects/ESpecialidadMachineLearning/Modulo3EvalMod_MarcoParra
collected 3 items                                                                                                            

tests/test_excepciones.py::test_matriz_singular_error PASSED                                                           [ 33%]
tests/test_excepciones.py::test_datos_insuficientes_error PASSED                                                       [ 66%]
tests/test_excepciones.py::test_parametros_no_convergentes_error PASSED                                                [100%]

===================================================== 3 passed in 0.05s =====================================================
```

### 🧪 ¿Qué valida cada test?

| Test                                    | Descripción                                                                                   |
| --------------------------------------- | --------------------------------------------------------------------------------------------- |
| `test_matriz_singular_error`            | Verifica que se lanza una excepción cuando la matriz $X^TX$ no puede invertirse.              |
| `test_datos_insuficientes_error`        | Simula explícitamente la falta de datos y espera que se levante la excepción correspondiente. |
| `test_parametros_no_convergentes_error` | Fuerza una divergencia en Gradient Descent con una tasa de aprendizaje exagerada.             |


---

## 🧑‍💻 Autor

**Marco Antonio Fernández Parra**
Especialidad en Machine Learning – Módulo 3
Proyecto Evaluación Modular – Regresión Lineal
Junio 2025

---



