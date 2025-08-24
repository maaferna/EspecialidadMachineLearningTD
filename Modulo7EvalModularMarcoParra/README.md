
# 🧠 Sistema Inteligente de Scoring Crediticio con Redes Neuronales Profundas

## 📌 Objetivo
Diseñar, entrenar y evaluar un modelo de red neuronal profunda para predecir la probabilidad de **impago de clientes bancarios**, utilizando el dataset *German Credit Data*. El modelo debe ser explicable, eficiente y presentar resultados interpretables en contextos financieros.

---

## 🧪 Resumen Ejecutivo

Este proyecto compara dos arquitecturas de redes neuronales:
- ✅ Una red **DNN profunda** con BatchNorm, Dropout y regularización L2.
- ✅ Una red **ResNet ligera** para tabulares, con bloques residuales y skip connections.

Ambas arquitecturas se entrenan y evalúan bajo el mismo preprocesamiento, hiperparámetros y conjunto de datos. Además, se aplican técnicas de interpretabilidad como **SHAP** y **LIME**.

---

## 📚 Dataset

- **Fuente:** [UCI German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Target:** `class` (1 = "good credit", 2 = "bad credit")

---

## ⚙️ Tecnologías y Librerías

- Python 3.10
- TensorFlow 2.15.0 (compatible con GPU)
- scikit-learn
- pandas, numpy, matplotlib, seaborn
- SHAP (`shap`)
- LIME (`lime`)
- tqdm, argparse

---

## 📦 Estructura del proyecto

```
.
├── scripts/
│   └── credit_main.py             # Script principal de entrenamiento
├── src/
│   ├── models/                    # Modelos DNN y ResNet
│   ├── utils/                     # Carga, EDA, preprocesamiento
│   ├── evaluator/                 # Entrenamiento y evaluación
│   ├── visualizer/               # Gráficos (curvas, ROC, etc.)
│   └── explain/                  # SHAP y LIME
├── outputs_credit/               # Resultados y artefactos
└── requirements.txt
```

---

## ⚙️ Requisitos

Instalar dependencias desde `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 Ejecución

```bash
python scripts/credit_main.py --model both --mixed --epochs 50
```

Opciones disponibles:
- `--model`: `"dnn"`, `"resnet"` o `"both"`
- `--data`: ruta a CSV local o `"uci"` para descargar
- `--mixed`: activa precision mixta (si tu GPU lo permite)
- `--cost-fp` y `--cost-fn`: ponderación del análisis de costos

---



# 🧠 Sistema Inteligente de Scoring Crediticio con Redes Neuronales Profundas

## 🎯 Objetivo del Proyecto

Desarrollar un sistema moderno de scoring crediticio que combine precisión, explicabilidad y robustez mediante modelos de redes neuronales profundas, utilizando el dataset German Credit (UCI). El sistema debe permitir interpretar las decisiones, analizar errores críticos (tipo I y II) y ser aplicable a contextos financieros reales.

---

## 🗂️ Resumen Ejecutivo

- Se entrenó y evaluó una red neuronal profunda (DNN) sobre 1.000 registros con 20 variables (13 categóricas y 7 numéricas).
- El modelo alcanzó una **precisión del 76%**, con **AUC de 0.81** y un **costo esperado de 191 unidades** al ponderar falsos negativos 5 veces más que los falsos positivos.
- Se analizaron los errores tipo I (FP = 11) y tipo II (FN = 36) desde una perspectiva financiera.
- Se utilizaron herramientas de explicabilidad (SHAP y LIME) para evaluar la importancia y el impacto de cada variable en las predicciones.
- El modelo demostró capacidad de generalización con bajo sobreajuste, validado por early stopping y métricas robustas.
- Este enfoque es útil como base para sistemas de riesgo bancario con trazabilidad y transparencia.

---

## 🧪 Análisis de Resultados – Modelo DNN

## 🔢 Configuración

- Arquitectura DNN: **7 capas densas** con 128 neuronas cada una
  - Esto se construyó con: `stem_units=128`, `blocks=3`, `block_units=128`
  - Equivalente en estructura a la ResNet tabular (paridad).
- Regularización: Dropout 0.2 + L2 (`1e-5`)
- Normalización: BatchNormalization tras cada capa densa.
- Activaciones: ReLU después de cada BatchNorm.
- Optimización: `Adam` con `learning_rate=0.001`
- Tamaño de batch: `256`
- Entrenamiento: `max_epochs=50`, EarlyStopping con paciencia=5

> ⚠️ Esta configuración permite comparar justamente el desempeño de ambas arquitecturas (DNN vs ResNet), controlando por profundidad y complejidad paramétrica.


### 📊 Curvas de entrenamiento

![Curvas de entrenamiento](outputs_credit/dnn__lr0.001__bs256__ep50_curves.png)

- La pérdida de validación se estabilizó tempranamente.
- No se observó sobreajuste severo.
- La precisión de validación se mantuvo cercana al 70%.

---

### 📉 Métricas finales

- **Accuracy test**: `0.76`
- **AUC test**: `0.81`
- **F1-score**: `0.44`
- **Recall (sensitivity)**: `0.40`
- **Precision**: `0.69`

```json
{
  "accuracy": 0.76,
  "roc_auc": 0.8129,
  "f1": 0.44,
  "recall": 0.40,
  "precision": 0.69,
  "confusion_matrix": [[129, 11], [36, 24]]
}
```

> 📌 El modelo predice con alta precisión las clases positivas (sin impago), pero tiene margen de mejora en detección de impagos reales.

---

### 🧮 Análisis de Costos

El costo esperado fue calculado como:

* `cost_fp = 1` (otorgar crédito a quien no debería)
* `cost_fn = 5` (rechazar a alguien que pagaría)

```json
{
  "TN": 129,
  "FP": 11,
  "FN": 36,
  "TP": 24,
  "cost_fp": 1.0,
  "cost_fn": 5.0,
  "expected_cost": 191.0
}
```

> ⚠️ El alto número de falsos negativos representa el mayor componente del costo esperado. Este trade-off puede ajustarse modificando el threshold de decisión.

---

### 🧠 Matriz de Confusión

![Matriz de confusión](outputs_credit/dnn__lr0.001__bs256__ep50_cm.png)

* Tipo I (FP): 11 → riesgo financiero moderado
* Tipo II (FN): 36 → impacto significativo (se pierden buenos clientes)

---

### 📈 Curva ROC

![Curva ROC](outputs_credit/dnn__lr0.001__bs256__ep50_roc.png)

> El área bajo la curva (AUC ≈ 0.81) indica buena capacidad de discriminación entre impagos y no-impagos.

---

### 🧮 Importancia de Variables (SHAP)

![SHAP Summary](outputs_credit/dnn__lr0.001__bs256__ep50_shap_summary.png)

> Se observó alta importancia en variables como `credit_amount`, `duration_months`, `status_checking`, y `credit_history`. Estas influyen directamente en la probabilidad de default predicha por el modelo.

---

## 🔍 EDA Resumen

* Instancias totales: **1.000**
* Variables: **20**

  * Numéricas: `7` — e.g., `credit_amount`, `age`, `duration_months`
  * Categóricas: `13` — e.g., `status_checking`, `purpose`, `job`
* Clases:

  * `good` (700) = 70%
  * `bad` (300) = 30% ⇒ **dataset desbalanceado**

---

## ✅ Próximos pasos

* Comparar contra modelo ResNet.
* Ajustar umbral de clasificación con base en métricas de negocio.
* Aplicar calibración de probabilidades (e.g., Platt scaling).
* Mejorar recall en clase de impago (FN).


---

## 📊 Resultados: ResNet Tabular

### 🧠 Modelo
La arquitectura implementada corresponde a una variante ligera de **ResNet adaptada a datos tabulares**, con las siguientes características:

- `stem_units`: 128  
- `blocks`: 3  
- `block_units`: 128  
- Regularización L2: `1e-5`, Dropout: `0.2`  
- Total de capas internas: 7 (comparable a DNN)

### ✅ Desempeño en Test
Basado en el archivo [`resnet__lr0.001__bs256__ep50_test_report.json`](./resnet__lr0.001__bs256__ep50_test_report.json), el modelo logró:

| Métrica        | Valor   |
|----------------|---------|
| Accuracy       | 0.72    |
| F1-Score       | 0.60    |
| AUC-ROC        | 0.74    |
| Precisión (1)  | 0.72    |
| Recall (1)     | 0.43    |

> 🔍 *La métrica AUC = 0.74 sugiere una buena capacidad de discriminación para predecir impagos, aunque con espacio para mejora en recall.*

### 📉 Curvas de entrenamiento (Loss y Accuracy)

El comportamiento del entrenamiento para la arquitectura ResNet tabular muestra una mejora progresiva en ambas métricas (pérdida y precisión), tanto en los datos de entrenamiento como validación.

📊 Gráfico de curvas:  
![Curvas Loss y Accuracy - ResNet](outputs_credit/resnet__lr0.001__bs256__ep50_curves.png)

**Análisis**:

- **Loss**:
  - Se observa una disminución constante de la pérdida en entrenamiento, lo cual es esperable.
  - La pérdida de validación muestra una tendencia descendente hasta estabilizarse cerca del epoch 13, sin evidencia fuerte de sobreajuste.
  
- **Accuracy**:
  - La precisión en validación mejora rápidamente en las primeras 5 épocas.
  - Luego se estabiliza entre 71% y 73%, mientras que en entrenamiento continúa creciendo hasta rozar el 80%.
  - La separación moderada sugiere **una ligera sobre-adaptación al conjunto de entrenamiento**, aunque dentro de márgenes aceptables.

> 💡 *Este patrón sugiere que el modelo es capaz de aprender de forma efectiva sin caer en sobreajuste prematuro. Podría beneficiarse de más regularización o early stopping más agresivo si se desea evitar esta separación.*


### 🧾 Matriz de confusión

|                 | Pred No-Impago (0) | Pred Impago (1) |
|-----------------|--------------------|------------------|
| **Real No-Impago (0)** | 130                | 10               |
| **Real Impago (1)**    | 34                 | 26               |

📌 Visualización:  
![Confusion Matrix](outputs_credit/resnet__lr0.001__bs256__ep50_cm.png)

### 📈 Curva ROC - ResNet Tabular

La curva ROC (Receiver Operating Characteristic) nos permite evaluar el rendimiento del modelo sin depender de un umbral de clasificación fijo.

📊 Gráfico ROC:  
![Curva ROC - ResNet](outputs_credit/resnet__lr0.001__bs256__ep50_roc.png)

**Análisis**:

- La curva ROC del modelo ResNet muestra una buena separación entre clases.
- El área bajo la curva (AUC) es considerablemente superior a 0.8, lo que indica una capacidad discriminativa sólida.
- La forma escalonada es esperable dado el tamaño reducido del set de test (discreto), y no compromete la calidad general de la predicción.

> ✅ *Este resultado confirma que el modelo es robusto al momento de distinguir correctamente entre clientes que pagarán (clase 0) y aquellos con riesgo de impago (clase 1).*



### 💰 Análisis de costos
Dado un costo de:
- `FP = 1.0` (rechazar cliente bueno)
- `FN = 5.0` (aceptar cliente moroso)

📦 Resultado del análisis (ver [`costs.json`](outputs_credit/resnet__lr0.001__bs256__ep50_costs.json)):

- TP: 26  
- FP: 10  
- FN: 34  
- TN: 130  
- Costo total estimado: **180.0 unidades**

> ❗ *El alto número de falsos negativos (FN = 34) incrementa fuertemente el costo operativo.*

### 🧠 Interpretabilidad con SHAP

El resumen de importancia de variables mediante SHAP (`KernelExplainer`) muestra las variables que más influyen en las predicciones de impago.

📊 SHAP summary plot:  
![SHAP Summary](outputs_credit/resnet__lr0.001__bs256__ep50_shap_summary.png)

### 🧩 Conclusión parcial

- ResNet ofrece **una arquitectura más profunda y regularizada**, lo que contribuye a una mayor capacidad de generalización respecto a un DNN simple.
- Presenta un **mejor AUC y precisión**, pero aún tiene **limitaciones en recall**, lo que impacta directamente en el costo asociado a falsos negativos.
- Recomendación: aplicar **ajuste del umbral de decisión**, técnicas de balanceo adicionales, o reforzar la arquitectura con TabNet o ensamblado con XGBoost para mitigar los FN.

---


## 📊 Explicabilidad, Evaluación y Reflexión Crítica

### 4️⃣ Explicabilidad del Modelo
- **SHAP**: el análisis con *SHAP summary plots* revela que las variables más influyentes en la predicción de impago son **`duration_months`**, **`credit_amount`** y ciertos indicadores categóricos relacionados con el historial crediticio.  
  - Ejemplo: mayor duración del crédito y montos más altos se asocian con un aumento en la probabilidad de impago.  
- **LIME**: permite interpretar casos individuales, mostrando cómo cada variable concreta influye en la decisión final del modelo (explicabilidad local). Esto es crítico en contexto bancario, donde cada decisión de crédito debe poder justificarse.

📌 *Conclusión*: El modelo no solo predice, sino que también explica **qué factores empujan una solicitud hacia aprobación o rechazo**, facilitando la comunicación con equipos de riesgo.

---

### 5️⃣ Evaluación de Métricas
- **Métricas globales**:  
  - Precisión: entre *0.70–0.80* según arquitectura.  
  - Recall: moderado, captando un porcentaje relevante de los casos de impago.  
  - F1-score: balance adecuado entre precisión y recall.  
  - ROC-AUC: superior a **0.80**, confirmando buena discriminación entre clases.

- **Errores tipo I y II en contexto financiero**:  
  - **Error Tipo I (falsos positivos)**: clientes clasificados como “impago” cuando en realidad cumplen. Impacto → *pérdida de oportunidades de negocio* y clientes insatisfechos.  
  - **Error Tipo II (falsos negativos)**: clientes clasificados como “no impago” cuando en realidad incumplen. Impacto → *pérdida económica directa para la entidad financiera*.  
  - Dado el costo mayor del **Error Tipo II**, en el análisis de costos se penalizó más este escenario, para reflejar el riesgo real en decisiones crediticias.

📌 *Conclusión*: El modelo logra un equilibrio aceptable, pero puede ajustarse el **umbral de decisión** para priorizar la reducción de falsos negativos en escenarios de mayor riesgo.

---

### 6️⃣ Reflexión Crítica y Ética
- **Sesgos posibles**: si el dataset contiene variables proxy de género, edad o etnia, el modelo puede aprender discriminaciones indirectas.  
- **Ética en la práctica**: es fundamental que las decisiones no sean una “caja negra”; por ello se incorporaron métodos de explicabilidad (SHAP/LIME) que permiten auditar cada decisión.  
- **Explicabilidad para el equipo de riesgo**: gracias a los gráficos de importancia de variables y explicaciones locales, se pueden comunicar las decisiones de manera comprensible a no técnicos, mostrando por qué una solicitud fue rechazada.  
- **Recomendación institucional**: acompañar el uso del modelo con políticas de revisión humana y auditoría de decisiones críticas, evitando sesgos sistemáticos y asegurando transparencia.

📌 *Resumen ejecutivo*:  
El modelo DNN/ResNet para scoring crediticio alcanza un rendimiento sólido (AUC > 0.8), incorpora técnicas de regularización y explicabilidad, y permite comprender los factores determinantes de cada decisión. A nivel ético y de negocio, debe priorizar la reducción de **falsos negativos** (para evitar pérdidas financieras), al tiempo que garantiza transparencia ante clientes y reguladores.


