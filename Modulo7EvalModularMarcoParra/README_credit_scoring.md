
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

¿Deseas que continúe con la sección de resultados, gráficas generadas y análisis interpretativo?
