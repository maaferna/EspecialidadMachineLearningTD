# 🌸 Proyecto: Transfer Learning con ResNet50 y EfficientNetB0

## 📌 Objetivo

Aplicar **Transfer Learning** con arquitecturas modernas (ResNet50 y EfficientNetB0) para resolver tareas de **clasificación de imágenes reales**.
El flujo completo incluye:

1. Carga y preprocesamiento de datos (`flower_photos` y `CIFAR-10`).
2. Adaptación del modelo base (capas adicionales).
3. Entrenamiento y validación.
4. Evaluación cuantitativa y visualización de resultados.

---

## 📝 Resumen Ejecutivo

Este proyecto demuestra cómo aprovechar modelos preentrenados en ImageNet para tareas de clasificación de imágenes en datasets reales.
Se compararon dos arquitecturas:

* **ResNet50** sobre el dataset de **flores**.
* **EfficientNetB0** sobre **CIFAR-10**.

Resultados destacados:

* ResNet50 alcanzó una **accuracy de validación >91%** en *flower\_photos*.
* EfficientNetB0 logró **\~92% en validación** en *CIFAR-10*.
* La fine-tuning de ResNet50 mejoró la capacidad de generalización al permitir que capas profundas también ajustaran pesos.

---

## ⚙️ Instalación y Dependencias

### 1️⃣ Crear entorno

```bash
conda create -n tf_transfer python=3.10
conda activate tf_transfer
```

### 2️⃣ Instalar dependencias

```bash
pip install tensorflow matplotlib seaborn scikit-learn
```

### 3️⃣ Descargar datasets

* **flowers**:

```bash
bash scripts/get_flowers.sh
```

* **CIFAR-10** se descarga automáticamente desde `keras.datasets`.

---

## ▶️ Ejecución

### Entrenar con ResNet50 en *flowers*

```bash
python -m scripts.main --dataset flowers --base resnet50 --epochs 50 --batch-size 32 --img-size 224
```

### Entrenar con EfficientNetB0 en *CIFAR-10*

```bash
python -m scripts.main --dataset cifar10 --base efficientnetb0 --epochs 50 --batch-size 64 --img-size 224
```

### Fine-Tuning de ResNet50 (descongelar capas profundas)

```bash
python -m scripts.main --dataset flowers --base resnet50 --epochs 50 --batch-size 32 --img-size 224 --finetune
```

---

## 📊 Resultados

### 🔹 ResNet50 en *flowers*

* **Curvas de entrenamiento**
  ![ResNet50 Curves](outputs_tl/resnet50__flowers__img224__bs32__ep50_curves.png)

* **Matriz de confusión**
  ![ResNet50 Confusion Matrix](outputs_tl/resnet50__flowers__img224__bs32__ep50_cm.png)

* **Predicciones vs reales**
  ![ResNet50 Predictions](outputs_tl/resnet50__flowers__img224__bs32__ep50_pred_grid.png)

---

### 🔹 EfficientNetB0 en *CIFAR-10*

* **Curvas de entrenamiento**
  ![EfficientNet Curves](outputs_tl/efficientnetb0__cifar10__img224__bs64__ep50_curves.png)

* **Matriz de confusión**
  ![EfficientNet Confusion Matrix](outputs_tl/efficientnetb0__cifar10__img224__bs64__ep50_cm.png)

* **Predicciones vs reales**
  ![EfficientNet Predictions](outputs_tl/efficientnetb0__cifar10__img224__bs64__ep50_pred_grid.png)

---

### 🔹 Fine-Tuning ResNet50

* **Curvas de entrenamiento (Fine-Tune)**
  ![FT ResNet50 Curves](outputs_tl/resnet50__flowers__img224__bs32__ep50__ft_curves.png)

* **Matriz de confusión**
  ![FT ResNet50 CM](outputs_tl/resnet50__flowers__img224__bs32__ep50__ft_cm.png)

* **Predicciones vs reales**
  ![FT ResNet50 Predictions](outputs_tl/resnet50__flowers__img224__bs32__ep50__ft_pred_grid.png)

---

## 📌 Análisis Crítico

### ❓ ¿Por qué se eligió esa arquitectura?

* **ResNet50**: probada en visión por computadora, buen balance entre profundidad y eficiencia.
* **EfficientNetB0**: diseñada para escalar eficientemente parámetros y FLOPs, ideal para CIFAR-10.

### ⚠️ Principales desafíos

* **Preprocesamiento**: necesario adaptar el tamaño de las imágenes a 224x224 para ambos modelos.
* **Overfitting**: evidente en algunas curvas; mitigado con *data augmentation* y *fine-tuning*.
* **Computo**: entrenamiento completo requiere GPU con suficiente memoria.

### 🔧 Mejoras para producción

* Aplicar **regularización adicional** (Dropout, L2).
* **Fine-tuning progresivo** (descongelar capas en bloques).
* Implementar **monitorización en tiempo real** (TensorBoard).
