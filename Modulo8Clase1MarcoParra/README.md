# Sesión 1 — NLP clásico en textos clínicos

## 📌 Metodología
- Corpus: 10 notas clínicas simuladas (una por línea).
- Preprocesamiento:
  - Conversión a minúsculas.
  - Eliminación de puntuación.
- Representación:
  - **BoW (Bag of Words)** y **TF-IDF** con n-grams.
- Parámetros explorados:
  - **n-grams (1,2):** unigrama + bigrama.
  - **n-grams (1,3):** unigrama + bigrama + trigramas.
- Medida de similitud: **coseno** entre documentos.
- Visualizaciones:
  - Heatmap de similitud.
  - Gráficos de los 10 términos más relevantes por documento.

---

## 📊 Resultados comparativos

### 🔹 Configuración 1: n-grams (1,2)
- Los bigramas capturan combinaciones simples como *“dolor abdominal”* o *“fiebre leve”*, pero aún aparecen tokens numéricos aislados (*“97”*, *“60”*).
- La matriz de similitud muestra relaciones débiles entre documentos, con valores en torno a 0.05–0.15.
- Fortalezas:
  - Sencillo, rápido de calcular.
  - Palabras frecuentes + algunas expresiones cortas relevantes.
- Limitaciones:
  - Mucho “ruido” de números sin contexto.
  - Similaridad baja → los documentos parecen más independientes de lo esperado.

### 🔹 Configuración 2: n-grams (1,3)
- La incorporación de trigramas mejora la captación de contexto:
  - Ejemplo: *“paciente masculino 45”*, *“cefalea dolor de”*, *“temperatura 37 se”*.
- Menor ruido de números aislados; los números tienden a asociarse con su unidad clínica (*“70 años”*, *“TA 130 80”*).
- La matriz de similitud muestra valores ligeramente más altos y consistentes.
- Fortalezas:
  - Expresiones clínicas completas (trigramas).
  - Mayor interpretabilidad de los términos clave.
- Limitaciones:
  - Aumenta la dimensionalidad del espacio.
  - Requiere corpus un poco más grande para aprovechar plenamente.

---

## 🧾 Interpretación final
- **n-grams (1,2):** útiles para una visión básica, pero capturan ruido en forma de números sueltos.
- **n-grams (1,3):** entregan términos más semánticos y clínicamente coherentes, lo que enriquece la interpretación de similitud.
- **Conclusión:**  
  Para textos clínicos breves, los trigramas aportan valor al contexto (edad + síntoma + condición). Sin embargo, la elección depende del tamaño del corpus:  
  - Corpus pequeño → (1,2) más estable.  
  - Corpus más amplio → (1,3) permite explotar expresiones clínicas compuestas.

---
