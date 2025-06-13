# Análisis de Algoritmos de Búsqueda y Notación Big O

## 🌍 Objetivo

Comparar el rendimiento de dos algoritmos de búsqueda (lineal y binaria) aplicados sobre listas de diferentes tamaños, midiendo su eficiencia temporal y visualizando sus comportamientos con respecto a la complejidad computacional.

---

## ⚙️ Algoritmos Implementados

### 1. `busqueda_lineal(lista, objetivo)`

* Recorre secuencialmente cada elemento de la lista.
* **Complejidad temporal**: O(n)
* **Uso recomendado**: listas no ordenadas o listas muy pequeñas.

### 2. `busqueda_binaria(lista, objetivo)`

* Requiere que la lista esté ordenada.
* Divide la lista por la mitad en cada paso.
* **Complejidad temporal**: O(log n)
* **Uso recomendado**: listas grandes y ordenadas.

---

## 📆 Escenarios de Evaluación

Se ejecutaron pruebas con los siguientes valores objetivo:

* ✔️ Inicio de la lista (mejor caso para búsqueda lineal)
* ✔️ Elemento en el medio
* ✔️ Elemento inexistente (peor caso)

Y con estos tamaños de lista:

* 10.000 elementos
* 100.000 elementos
* 1.000.000 elementos

---

## ⏱️ Resultados de Tiempo (segundos)


### 🔎 **Resumen de Complejidad Teórica**

| Algoritmo            | Complejidad Temporal | Descripción                                                                    |
| -------------------- | -------------------- | ------------------------------------------------------------------------------ |
| **Búsqueda Lineal**  | O(n)                 | Tiempo crece linealmente con el tamaño de la lista.                            |
| **Búsqueda Binaria** | O(log n)             | Tiempo crece logarítmicamente (muy lento en aumento). Requiere lista ordenada. |

---

### ✅ **Análisis por Tamaño**

#### 1. Tamaño **10.000**

* **Lineal**:

  * `inicio`: \~1.4e-6 → extremadamente rápido porque está en la primera posición.
  * `medio`: \~0.000127 → tarda más, porque recorre \~5.000 elementos.
  * `no_existe`: \~0.00024 → el peor caso (revisa los 10.000 elementos).

* **Binaria**:

  * Todos los tiempos están cerca de 2-3 microsegundos → esperable, ya que `log₂(10.000) ≈ 14` operaciones.

#### 2. Tamaño **100.000**

* **Lineal**:

  * `inicio`: \~1e-6 (igual de rápido).
  * `medio`: \~0.00124 → más tiempo que antes (10 veces más elementos que en 10.000).
  * `no_existe`: \~0.0024 → casi proporcional al tamaño.

* **Binaria**:

  * Se mantiene en \~2.3 microsegundos → log₂(100.000) ≈ 17 → casi no crece.

#### 3. Tamaño **1.000.000**

* **Lineal**:

  * `inicio`: \~4.4e-6 → igual de rápido.
  * `medio`: \~0.0125 → 10 veces más que con 100.000, esperado.
  * `no_existe`: \~0.025 → como debe ser (recorre todo).

* **Binaria**:

  * `inicio`: \~1.2e-5 → un poco más alto (posiblemente por más profundidad en las llamadas).
  * `medio`: \~7.8e-6
  * `no_existe`: \~3.1e-6 → muy bajo, lo cual concuerda con el hecho de que los tiempos de búsqueda binaria apenas cambian con el tamaño.

---

### 📊 Conclusión

Los resultados **están correctos** y son **consistentes con la teoría**:

* Búsqueda **lineal** escala con `O(n)` — tiempos aumentan proporcionalmente con el tamaño.
* Búsqueda **binaria** escala con `O(log n)` — tiempos casi constantes, solo aumentan levemente.
* En todos los casos, la binaria es **mucho más eficiente**, especialmente para listas grandes y búsquedas hacia el final o inexistentes.


---

## 🔍 Análisis

### Búsqueda Lineal

* Escanea todos los elementos uno por uno.
* Se ve afectada directamente por el tamaño de la lista.
* En el peor caso (elemento no existente), el tiempo crece linealmente.

### Búsqueda Binaria

* Eficiente incluso con millones de elementos.
* Su tiempo se mantiene casi constante al crecer el tamaño (comportamiento logarítmico).

---

## 🔢 Visualización del Rendimiento

El siguiente gráfico muestra la comparación de tiempo para cada método:

![Gráfico de Comparación](screenshots/grafico_comparacion_busqueda.png)

---

## ✍️ Conclusión

* La **búsqueda binaria** es claramente superior en eficiencia, pero **solo puede usarse si la lista está ordenada**.
* La **búsqueda lineal** es simple y flexible, pero se vuelve ineficiente en listas grandes.

### ✅ Recomendaciones

* Usar **búsqueda binaria** en entornos de datos estructurados, especialmente si el rendimiento es clave.
* Reservar **búsqueda lineal** para casos de datos no ordenados o listas pequeñas.

---

## 📄 Archivos Generados

* `main.py`: ejecución principal del análisis
* `utils.py`: funciones auxiliares y algoritmos de búsqueda
* `grafico_comparacion_busqueda.png`: gráfico de tiempos

---

## 📚 Bibliotecas Utilizadas

* `timeit` para medición de rendimiento
* `matplotlib` para visualización

---

> Proyecto desarrollado como parte del módulo: **Análisis de Algoritmos y Notación Big O**
