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

📊 **Matriz de confusión (test set)**  
![Confusion matrix](outputs_imdb/imdb_lstm__adam2e-4_clip1.0__bce__vs0.2__bs64__ep10_confusion_matrix.csv)

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

## ✅ Conclusión
El modelo LSTM implementado cumple los objetivos de la actividad:
- Aprende representaciones secuenciales de las reseñas.
- Generaliza en el rango esperado (~86% en test).
- Evidencia limitaciones para manejar memorias largas y prevenir sobreajuste.  

Esto aporta evidencia de la **dificultad de “recordar” secuencias completas**, lo cual justifica la evolución hacia arquitecturas más avanzadas (como Transformers) en procesamiento de lenguaje natural.
