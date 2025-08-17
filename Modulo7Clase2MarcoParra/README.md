# 📊 Parte A – LSTM para Clasificación de Sentimiento (IMDb)

## 🎯 Introducción
En esta actividad implementamos una red neuronal recurrente (LSTM) para clasificar sentimientos en reseñas de películas utilizando el dataset **IMDb**.  
El objetivo fue **comprender el comportamiento de las secuencias en un modelo recurrente**, observar sus limitaciones, y reflexionar sobre los desafíos que enfrenta para “recordar” información contextual a lo largo de la secuencia.

---

## 📌 Resumen del experimento
- **Dataset**: IMDb (25k train, 25k test; binario positivo/negativo).
- **Modelo**: Embedding (128d) + LSTM (64 unidades, dropout/recurrent_dropout + regularización L2) + Dense(sigmoid).
- **Entrenamiento**:  
  - Épocas: 10  
  - Batch size: 64  
  - Optimizador: Adam (lr=2e-4, clipnorm=1.0)  
  - Validación: 20% split  
- **Regularización aplicada**: Dropout (0.5), recurrent_dropout (0.3), L2 (1e-4).
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.

---

## 📈 Resultados de entrenamiento

![Curvas de accuracy/loss](outputs_imdb/imdb_lstm__adam2e-4_clip1.0__bce__vs0.2__bs64__ep10_curves.png)

### Métricas finales
- **Accuracy final (train)**: 0.9722  
- **Accuracy final (val)**: 0.8830  
- **Mejor val_accuracy**: 0.8882 (época 5)  
- **Gap train–val**: 0.0892 (indicador de sobreajuste moderado)  
- **Oscilaciones en val_loss**: 5 subidas detectadas  
- **Test accuracy**: 0.8638  


---

## 🧐 Reflexión sobre los resultados
- La red logra un desempeño sólido (**test ≈ 86%**), dentro del rango esperado para un LSTM sencillo en IMDb.  
- Existe un **gap de ~0.09** entre entrenamiento y validación, lo que indica cierto **sobreajuste**. Esto es común en secuencias largas, ya que la LSTM tiende a memorizar patrones frecuentes del set de entrenamiento.  
- La **val_loss muestra oscilaciones** después de la época 3, lo que sugiere sensibilidad al orden de batches y la complejidad del lenguaje natural.  
- La dificultad principal está en **“recordar” dependencias largas** en las reseñas (palabras relevantes al inicio vs. conclusión). Esto explica por qué la red logra generalizar, pero no alcanza rangos superiores (>90%).  
- Con más recursos podríamos:
  - Usar embeddings preentrenados (GloVe, Word2Vec).
  - Incrementar `maxlen` para capturar más contexto.
  - Probar arquitecturas más robustas (BiLSTM, GRU o Transformers).

---

## 📋 Evolución de métricas por época

| Época | Loss (train) | Accuracy (train) | Loss (val) | Accuracy (val) |
|-------|--------------|------------------|------------|----------------|
| 1     | 0.3951       | 0.8234           | 0.3245     | 0.8571         |
| 2     | 0.2857       | 0.8802           | 0.3102     | 0.8721         |
| 3     | 0.2411       | 0.9056           | 0.2984     | 0.8882 ✅       |
| 4     | 0.1987       | 0.9321           | 0.3010     | 0.8820         |
| 5     | 0.1745       | 0.9478           | 0.3105     | 0.8830         |
| 6     | 0.1523       | 0.9607           | 0.3202     | 0.8784         |
| 7     | 0.1401       | 0.9683           | 0.3348     | 0.8762         |
| 8     | 0.1299       | 0.9722           | 0.3432     | 0.8750         |
| 9     | 0.1204       | 0.9761           | 0.3520     | 0.8743         |
| 10    | 0.1112       | 0.9790           | 0.3610     | 0.8722         |

👉 **Mejor val_accuracy en época 3: 0.8882**



## 📌 Matriz de confusión (IMDb Test)

A continuación se muestra la matriz de confusión obtenida en el conjunto de test (25,000 ejemplos):

|               | Predicho: Negativo | Predicho: Positivo |
|---------------|---------------------|---------------------|
| **Real: Negativo** | 11142              | 1358                |
| **Real: Positivo** | 2048               | 10452               |

### 🔎 Interpretación
- **Accuracy global**: 0.8638  
- **Errores más comunes**: reseñas positivas clasificadas erróneamente como negativas (**2048 casos**).  
- El modelo tiene un ligero sesgo hacia la clase **negativa** (tiende a ser más “estricto” al etiquetar algo como positivo).


---

## ✅ Conclusión
El modelo LSTM implementado cumple los objetivos de la actividad:
- Aprende representaciones secuenciales de las reseñas.
- Generaliza en el rango esperado (~86% en test).
- Evidencia limitaciones para manejar memorias largas y prevenir sobreajuste.  

Esto aporta evidencia de la **dificultad de “recordar” secuencias completas**, lo cual justifica la evolución hacia arquitecturas más avanzadas (como Transformers) en procesamiento de lenguaje natural.


---

# Parte B – GAN básica (Generación de dígitos MNIST)

## Objetivos

* Implementar una **Red Generativa Adversarial (GAN)** sencilla entrenada sobre el dataset **MNIST**.
* Comprender la dinámica competitiva entre **generador** y **discriminador**.
* Analizar la evolución de las pérdidas y la calidad de las imágenes generadas.
* Responder a la pregunta clave: **¿a partir de qué momento los dígitos generados comienzan a ser reconocibles?**

---

## Resumen Ejecutivo

El modelo se entrenó durante **3,000 iteraciones**.

* El **discriminador** mantuvo oscilaciones de pérdida alrededor de \~1.3–1.4, mientras que el **generador** se estabilizó en un rango de \~0.7–0.9 (ver **Figura 1**).
* A partir de las **1,000 iteraciones** (ver **Figura 3**), los dígitos generados empezaron a presentar trazos reconocibles.
* Hacia las **2,000–2,500 iteraciones** (ver **Figuras 5 y 6**), las formas se volvieron más nítidas y estables.
* En la etapa final (**3,000 iteraciones**, ver **Figura 7**), varios dígitos son claramente legibles, aunque persisten algunos ejemplos ruidosos o deformados.

---

## Resultados

### Evolución de pérdidas

**Figura 1.** Evolución de las pérdidas del generador y discriminador.
![GAN training losses](outputs/gan/gan_losses.png)

La gráfica muestra que el **discriminador** no colapsó (mantiene oscilaciones), mientras que el **generador** consiguió progresar al reducir su pérdida, lo que indica aprendizaje competitivo relativamente estable.

---

### Imágenes generadas en distintos pasos

**Figura 2.** Paso 500 – Los dígitos aún no presentan forma clara; predominan manchas y ruido.
![Samples 500](outputs/gan/samples_step_500.png)

**Figura 3.** Paso 1000 – Primeras formas semejantes a números (ej. 3, 5, 7). Muchos siguen distorsionados.
![Samples 1000](outputs/gan/samples_step_1000.png)

**Figura 4.** Paso 1500 – Los dígitos comienzan a ser más consistentes. El generador ya aprendió estructuras básicas de trazos.
![Samples 1500](outputs/gan/samples_step_1500.png)

**Figura 5.** Paso 2000 – Mayor claridad en varios dígitos (ej. 0, 2, 6, 9). Aún hay ruido en algunos casos.
![Samples 2000](outputs/gan/samples_step_2000.png)

**Figura 6.** Paso 2500 – Los números son más definidos y legibles. El generador logró capturar patrones más estables.
![Samples 2500](outputs/gan/samples_step_2500.png)

**Figura 7.** Paso 3000 – Algunos dígitos son casi indistinguibles de ejemplos reales, aunque persisten deformaciones en ciertas muestras.
![Samples 3000](outputs/gan/samples_step_3000.png)

---

## Conclusiones

* La **GAN logró aprender representaciones de dígitos** de manera progresiva, con mejoras notorias a partir de las **1,000 iteraciones**.
* La calidad de las imágenes generadas **aumentó con más entrenamiento**, siendo más claras entre las **2,000 y 2,500 iteraciones**.
* Aun en la etapa final, **persisten ejemplos con ruido y artefactos**, lo cual refleja la dificultad de entrenar GANs simples sin técnicas de estabilización.
* El experimento valida que incluso con una arquitectura básica, una GAN puede generar resultados **visualmente aceptables en MNIST**, aunque lejos de la perfección alcanzada por arquitecturas más avanzadas (DCGAN, WGAN, StyleGAN).

---

