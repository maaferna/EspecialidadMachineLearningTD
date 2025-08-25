# 🔎 Resultados Autoencoders (MNIST)

## 📌 Objetivo
Comparar el desempeño de:
- **Autoencoder básico (AE)**: reconstrucción directa de imágenes MNIST.
- **Autoencoder con denoising (DAE)**: entrenado con ruido gaussiano (σ=0.5) para robustez.

El fin es analizar cómo la adición de ruido afecta la reconstrucción, la estabilidad del entrenamiento y la capacidad de generalización del modelo.

---

## 📊 Resultados de entrenamiento

### AE básico
- **Curvas de entrenamiento**  
  ![Curvas Basic AE](outputs_ae/ae_basic__lr0.001__bs128__ep50_curves.png)

- **Reconstrucciones**  
  ![Reconstrucciones Basic AE](outputs_ae/ae_basic__lr0.001__bs128__ep50_recons.png)

- **Métricas finales (época 50)**  
  - Train loss: ~0.0149  
  - Val loss: ~0.0095  
  - MAE val: ~0.032  

✔ Buen ajuste y convergencia estable.  
⚠ Posible **sobreajuste leve**: la brecha train/val se reduce pero no desaparece.

---

### AE con Denoising (σ=0.5)
- **Curvas de entrenamiento**  
  ![Curvas Denoising AE](outputs_ae/ae_denoise__lr0.001__bs128__ep50__sigma0.5_curves.png)

- **Reconstrucciones (denoising)**  
  ![Reconstrucciones Denoising AE](outputs_ae/ae_denoise__lr0.001__bs128__ep50__sigma0.5_denoise_recons.png)

- **Métricas finales (época 50)**  
  - Train loss: ~0.0204  
  - Val loss: ~0.0172  
  - MAE val: ~0.047  

✔ Logra **recuperar estructura de dígitos aun con ruido fuerte**.  
⚠ Coste: mayor error absoluto (MAE ↑) respecto al AE básico, lo cual es esperado.

---

## 📈 Comparación

- **AE básico** alcanza menor error de reconstrucción absoluto.  
- **DAE** sacrifica exactitud pixel a pixel, pero **generaliza mejor a datos corruptos**, manteniendo legibilidad de dígitos.  
- En tareas prácticas (ej. preprocesamiento o reducción de ruido en imágenes reales), el **DAE es preferible**.

---

## 🤔 Reflexión
1. **Eficiencia computacional**: ambos entrenan rápido con batch=128 en GPU (4090), ocupando ~2–3 GB VRAM.  
2. **Robustez**: el denoising AE demuestra la importancia de introducir ruido en entrenamiento para entornos con señales degradadas.  
3. **Siguientes pasos**:
   - Extender a **Convolutional Autoencoders (Conv-AE)** para aprovechar la estructura espacial.  
   - Explorar **variational autoencoders (VAE)** para generación sintética.  
   - Ajustar **σ dinámico** (curriculum de ruido) para mejorar robustez sin sacrificar tanto MAE.

---

# 🔎 Resultados Autoencoders — Comparación 20 vs 50 épocas

## 📌 Objetivo
Evaluar cómo cambia el desempeño de un **Autoencoder básico (AE)** y un **Autoencoder con denoising (DAE, σ=0.5)** cuando se entrenan por **20 vs. 50 épocas**, en términos de reconstrucción, convergencia y robustez al ruido.

---

## 📊 Resultados AE Básico

### 20 épocas
- **Curvas de entrenamiento**  
  ![Curvas Basic AE 20](outputs_ae/ae_basic__lr0.001__bs128__ep20_curves.png)

- **Reconstrucciones**  
  ![Reconstrucciones Basic AE 20](outputs_ae/ae_basic__lr0.001__bs128__ep20_recons.png)

- **Métricas finales (época 20)**  
  - Val loss: ~0.0172  
  - Val MAE: ~0.0478  

✔ Aprendizaje rápido en pocas épocas.  
⚠ Menor precisión de reconstrucción comparado con 50 épocas.

---

### 50 épocas
- **Curvas de entrenamiento**  
  ![Curvas Basic AE 50](outputs_ae/ae_basic__lr0.001__bs128__ep50_curves.png)

- **Reconstrucciones**  
  ![Reconstrucciones Basic AE 50](outputs_ae/ae_basic__lr0.001__bs128__ep50_recons.png)

- **Métricas finales (época 50)**  
  - Val loss: ~0.0095  
  - Val MAE: ~0.032  

✔ Reconstrucciones mucho más nítidas.  
⚠ Indicios leves de sobreajuste (gap train/val).

---

## 📊 Resultados AE con Denoising (σ=0.5)

### 20 épocas
- **Curvas de entrenamiento**  
  ![Curvas Denoising AE 20](outputs_ae/ae_denoise__lr0.001__bs128__ep20__sigma0.5_curves.png)

- **Reconstrucciones**  
  ![Reconstrucciones Denoising AE 20](outputs_ae/ae_denoise__lr0.001__bs128__ep20__sigma0.5_denoise_recons.png)

- **Métricas finales (época 20)**  
  - Val loss: ~0.0172  
  - Val MAE: ~0.0473  

✔ Capacidad inicial de limpiar ruido.  
⚠ Denoise parcial: todavía borroso en algunos dígitos.

---

### 50 épocas
- **Curvas de entrenamiento**  
  ![Curvas Denoising AE 50](outputs_ae/ae_denoise__lr0.001__bs128__ep50__sigma0.5_curves.png)

- **Reconstrucciones**  
  ![Reconstrucciones Denoising AE 50](outputs_ae/ae_denoise__lr0.001__bs128__ep50__sigma0.5_denoise_recons.png)

- **Métricas finales (época 50)**  
  - Val loss: ~0.0172  
  - Val MAE: ~0.047  

✔ Reconstrucciones más consistentes que en 20 épocas.  
⚠ El error se mantiene mayor al AE básico, lo cual es esperado porque debe lidiar con ruido adicional.

---

## 📈 Comparación General

- **20 épocas**:  
  - AE básico ya reconstruye aceptablemente, pero el denoise aún muestra limitaciones.  
  - Ideal para entrenamiento rápido y prototipado.  

- **50 épocas**:  
  - AE básico logra **alta fidelidad visual** (menor MAE).  
  - DAE mantiene **robustez al ruido**, sacrificando algo de detalle.  
  - Ambos modelos alcanzan convergencia estable.  

---

## 🤔 Reflexión
- El **número de épocas impacta fuertemente** al AE básico (aprende más detalle con más tiempo).  
- En el **DAE**, la diferencia entre 20 y 50 épocas es más leve: el entrenamiento extra refina, pero el ruido siempre limita la reconstrucción perfecta.  
- **Aplicación práctica**:  
  - Para compresión pura → AE básico + mayor número de épocas.  
  - Para limpieza/robustez en datos ruidosos → DAE incluso con menos épocas.


---


# 🔍 Análisis Comparativo y Reflexión Final

## 📊 Comparación entre AE Básico y Denoising AE

En términos cuantitativos, el **Autoencoder básico (AE)** logra **pérdidas de reconstrucción menores** y un **MAE más bajo** (≈0.032 tras 50 épocas) en comparación con el **Denoising Autoencoder (DAE)** (≈0.047 con ruido σ=0.5). Esto se traduce en reconstrucciones más nítidas y detalladas en el AE clásico, especialmente cuando se entrena durante más épocas.  

En cambio, el **DAE** muestra reconstrucciones menos precisas desde el punto de vista visual, pero con la **ventaja de eliminar ruido** y mantener estructuras reconocibles en escenarios adversos. A nivel visual, el AE básico sobresale cuando los datos no contienen ruido significativo, mientras que el DAE ofrece mayor robustez y estabilidad frente a perturbaciones externas.  

En resumen:
- **AE Básico**: mejor fidelidad visual, menor pérdida de reconstrucción.  
- **DAE**: mejor tolerancia al ruido, aunque sacrifica detalles.  
- **Aplicación práctica**: la elección depende del contexto; en entornos controlados se prefiere AE, en entornos ruidosos o inseguros, DAE.

---

## 🤔 Reflexión sobre aplicaciones (≈230 palabras)

Los autoencoders, tanto básicos como de denoising, tienen un enorme potencial en **medicina, seguridad e industria**. En medicina, los **DAE** podrían utilizarse para mejorar la calidad de imágenes médicas ruidosas como radiografías, resonancias magnéticas o tomografías, permitiendo diagnósticos más confiables incluso en condiciones de baja calidad de captura. El **AE básico**, por otro lado, puede emplearse en compresión de imágenes médicas, reduciendo almacenamiento y costos de transmisión de datos sin perder información clínica crítica.

En el ámbito de la **seguridad**, los autoencoders permiten detectar anomalías en video vigilancia. Por ejemplo, un modelo entrenado con grabaciones de entornos normales puede identificar actividades sospechosas al detectar desviaciones en la reconstrucción. En ciberseguridad, pueden emplearse para identificar patrones inusuales en tráfico de red, ayudando a prevenir ataques.

En la **industria**, estas técnicas facilitan mantenimiento predictivo. Un AE entrenado con señales de máquinas en buen estado puede señalar fallas incipientes cuando la reconstrucción del estado real diverge de lo esperado. Además, el denoising AE puede limpiar señales de sensores industriales ruidosos, mejorando la confiabilidad en tiempo real.

En conclusión, mientras que el **AE básico** es ideal para compresión y representación eficiente, el **DAE** aporta robustez y capacidad de operar en ambientes adversos. Ambos modelos, correctamente aplicados, son herramientas valiosas que pueden transformar la forma en que distintas áreas procesan y entienden la información.


