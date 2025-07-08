
# 🧮 Resolución de Sistemas de Ecuaciones y Transformaciones 2D

Este proyecto implementa la resolución de sistemas de ecuaciones lineales (cuadrados, sobredeterminados y subdeterminados) utilizando **álgebra matricial en Python**, así como la aplicación de **transformaciones lineales 2D** (rotación + escalado) para representar visualmente su efecto sobre un conjunto de puntos.

---

## 📁 Estructura del Proyecto


Modulo3Clase2/
├── scripts/
│   └── main.py                      # Punto de entrada del sistema
├── src/
│   ├── algebra\_solver.py           # Resolución de sistemas con np.linalg
│   ├── sistemas\_lineales.py        # Definición y ejecución de sistemas lineales
│   ├── transformador\_2d.py         # Funciones de rotación, escalado y aplicación
│   ├── utils\_plot.py               # Funciones gráficas (barras y transformaciones)
│   └── utils.py                    # Funciones de guardado y ejecución de transformaciones por sistema
├── outputs/
│   ├── solucion\_*.json             # Resultados guardados por tipo de sistema
│   └── grafico\_solucion\_*.png      # Barras con soluciones
│   └── transformacion\_\*.png        # Gráficos de transformaciones por sistema


---

## ⚙️ Funcionalidades

### 🔹 Resolución de Sistemas

- Se definen tres tipos de sistemas:
  - **Cuadrado:** Matriz A 3x3, vector b de tamaño 3.
  - **Sobredeterminado:** A 3x2, b de tamaño 3 (más ecuaciones que incógnitas).
  - **Subdeterminado:** A 2x3, b de tamaño 2 (más incógnitas que ecuaciones).
- Se utilizan funciones de álgebra lineal de `NumPy`:
  - `np.linalg.solve()` para sistemas cuadrados.
  - `np.linalg.lstsq()` para los demás, obteniendo soluciones por mínimos cuadrados.

### 🔹 Transformación Lineal 2D

- Se define una matriz compuesta `T = S @ R`:
  - `R`: Rotación de 45°.
  - `S`: Escalado uniforme por 1.5.
- Se aplica la transformación a un conjunto de puntos para **cada tipo de sistema**.
- Se grafican los puntos originales y transformados, guardando un gráfico por tipo.

---

## 🧠 Decisiones de Diseño

- **Modularización:** Cada aspecto del proyecto (resolución, visualización, transformaciones) está separado en archivos específicos, siguiendo principios de claridad y reutilización.
- **Reutilización de lógica:** Se crearon funciones parametrizables para soportar múltiples tipos de sistemas con el mismo flujo de resolución y graficación.
- **Compatibilidad con PEP 8:** Todos los archivos incluyen docstrings y cumplen las normas de estilo para facilitar mantenimiento y comprensión.
- **Visualización clara:** Se generan salidas gráficas que conectan el análisis numérico con una representación visual de sus efectos.

---

## 📦 Dependencias
conda environment.yml file
- Python 3.8+
- NumPy
- Matplotlib

---

## 🚀 Ejecución

```bash
python scripts/main.py
````

Esto:

* Resuelve los tres sistemas.
* Guarda resultados en formato JSON.
* Genera gráficos de barras y transformaciones en la carpeta `outputs`.

---

## 👨‍🏫 Uso Educativo

Este proyecto está diseñado como una actividad práctica para reforzar el uso de álgebra matricial en Python, el entendimiento geométrico de transformaciones lineales, y el desarrollo de código limpio y bien documentado en proyectos de ingeniería o ciencia de datos.

