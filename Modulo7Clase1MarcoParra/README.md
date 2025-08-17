# Clasificador Fashion-MNIST — MLP (Curso)

Proyecto de clasificación de imágenes (28×28) del dataset **Fashion-MNIST** con una red neuronal **completamente conectada** (MLP), siguiendo el estilo del curso.

## 🎯 Objetivo
- Cargar y preprocesar Fashion-MNIST (normalización [0,1] + one-hot).
- Diseñar un MLP: `Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`.
- Entrenar con `optimizer='adam'`, `loss='categorical_crossentropy'`, `epochs=10`, `batch_size=32`, `validation_split=0.2`.
- Evaluar en test y visualizar curvas de **loss** y **accuracy**.

---

## 📦 Estructura del proyecto
```
scripts/
main.py
src/
utils/{preprocessing.py, data\_loader.py}
models/mlp.py
evaluator/train\_eval.py
visualizer/plots.py
outputs/
course\_mlp\_fashion\_\_adam\_\_cce\_\_vs0.2\_\_bs32\_\_ep10\_curves.png
course\_mlp\_fashion\_\_adam\_\_cce\_\_vs0.2\_\_bs32\_\_ep10\_history.json
course\_mlp\_fashion\_\_adam\_\_cce\_\_vs0.2\_\_bs32\_\_ep10\_history\_qc.md
course\_mlp\_fashion\_\_adam\_\_cce\_\_vs0.2\_\_bs32\_\_ep10\_test\_report.json
course\_mlp\_fashion\_\_adam\_\_cce\_\_vs0.2\_\_bs32\_\_ep10\_model.keras

```

---

## ⚙️ Entorno y ejecución

### En Google Colab (recomendado)
1. **Runtime → Change runtime type → GPU**.
2. Monta Drive y apunta a la carpeta del proyecto:
   ```python
   from google.colab import drive; drive.mount('/content/drive', force_remount=True)
   PROJECT_ROOT = "/content/drive/MyDrive/Modulo7Clase1MarcoParra"  
   # ajusta tu ruta
  ```

3. Instala deps mínimas:

  ```python
  !pip -q install "tensorflow==2.19.0" "tf-keras==2.19.0" "tensorflow-text==2.19.0" "tensorflow-decision-forests==1.12.0" "numpy==2.0.2" matplotlib pillow
   ```

4. Añade `PROJECT_ROOT` al `PYTHONPATH` y ejecuta:

   ```python
   import os, sys
   sys.path.insert(0, PROJECT_ROOT); os.chdir(PROJECT_ROOT)
   from scripts import main as train_main
   train_main.main(out_dir="outputs")
   ```

### Local (conda)

```bash
conda env create -f environment.yml
conda activate especialidadmachinelearning
python -m scripts.main
```

---

## 🧠 Arquitectura del modelo

* `Flatten(input_shape=(28,28))`
* `Dense(128, activation='relu')`
* `Dense(64, activation='relu')`
* `Dense(10, activation='softmax')`

**Pérdida:** `categorical_crossentropy`
**Optimizador:** `adam`
**Métrica:** `accuracy`
**Épocas:** 10 — **Batch:** 32 — **Validation split:** 0.2

---

## 📈 Resultados
# Resultados — Fashion-MNIST (MLP, Grid)

Este informe resume el grid ejecutado sobre **Fashion-MNIST** con MLP:
- **Activaciones**: (ReLU, ReLU) y (ReLU, Tanh)  
- **Pérdidas**: `categorical_crossentropy`, `mse`  
- **Optimizadores**: `adam`, `sgd(momentum=0.9, lr=0.01)`  
- **Entrenamiento**: `epochs=10`, `batch_size=32`, `validation_split=0.2`  

---

## 🏆 Resumen ejecutivo

- **Mejor combinación (test)**: **MSE + Adam** con **test_acc = 0.8781**  
- **Activaciones**: **(ReLU, ReLU)**  
- **Mejor val_acc** en ese run: **0.8875**

> Ver detalle en: `outputs/reflection_enunciado.md`

---

## 📊 Comparativa general

**Gráfico comparativo (test accuracy por corrida):**

![Comparación de corridas](outputs/experiments_comparison.png)

---

## 📈 Tabla de resultados (val_best_acc y test_acc)

> Fuente: `outputs/experiments_summary.csv`

| Run | Activaciones | Pérdida | Optimizador | val_best_acc | test_acc |
|---|---|---:|---:|---:|---:|
| mlp_relu_relu__categorical_crossentropy__adam | (relu, relu) | categorical_crossentropy | adam | **0.8887** | **0.8706** |
| mlp_relu_relu__categorical_crossentropy__sgd  | (relu, relu) | categorical_crossentropy | sgd  | **0.8851** | **0.8641** |
| mlp_relu_relu__mse__adam                      | (relu, relu) | mse                     | adam | **0.8875** | **0.8781** ⭐ |
| mlp_relu_relu__mse__sgd                       | (relu, relu) | mse                     | sgd  | **0.8566** | **0.8490** |
| mlp_relu_tanh__categorical_crossentropy__adam | (relu, tanh) | categorical_crossentropy | adam | **0.8857** | **0.8760** |
| mlp_relu_tanh__categorical_crossentropy__sgd  | (relu, tanh) | categorical_crossentropy | sgd  | **0.8839** | **0.8763** |
| mlp_relu_tanh__mse__adam                      | (relu, tanh) | mse                     | adam | **0.8863** | **0.8741** |
| mlp_relu_tanh__mse__sgd                       | (relu, tanh) | mse                     | sgd  | **0.8601** | **0.8482** |

> ⭐ = **mejor test_acc**

---
## 🖼️ Curvas de entrenamiento (se ven aquí)

> Cada imagen muestra **loss** y **accuracy** de train/val para la corrida indicada.

### (ReLU, ReLU)
<p>
  <img src="outputs/mlp_relu_relu__categorical_crossentropy__adam_curves.png" alt="relu_relu CCE Adam" width="400"/>
  <img src="outputs/mlp_relu_relu__categorical_crossentropy__sgd_curves.png" alt="relu_relu CCE SGD" width="400"/>
</p>
<p>
  <img src="outputs/mlp_relu_relu__mse__adam_curves.png" alt="relu_relu MSE Adam" width="400"/>
  <img src="outputs/mlp_relu_relu__mse__sgd_curves.png" alt="relu_relu MSE SGD" width="400"/>
</p>

### (ReLU, Tanh)
<p>
  <img src="outputs/mlp_relu_tanh__categorical_crossentropy__adam_curves.png" alt="relu_tanh CCE Adam" width="400"/>
  <img src="outputs/mlp_relu_tanh__categorical_crossentropy__sgd_curves.png" alt="relu_tanh CCE SGD" width="400"/>
</p>
<p>
  <img src="outputs/mlp_relu_tanh__mse__adam_curves.png" alt="relu_tanh MSE Adam" width="400"/>
  <img src="outputs/mlp_relu_tanh__mse__sgd_curves.png" alt="relu_tanh MSE SGD" width="400"/>
</p>


---
## 🧾 Artefactos por corrida (con visualización embebida)

> Cada tarjeta muestra la **curva de entrenamiento** (loss/accuracy) y enlaces a **matriz de confusión (CSV)**, **history (JSON)**, **QC (MD)** y **reporte de test (JSON)**.  
> Todos los paths son relativos a `outputs/`.

### (ReLU, ReLU) — `categorical_crossentropy`

**Adam**
<p>
  <img src="outputs/mlp_relu_relu__categorical_crossentropy__adam_curves.png" alt="relu_relu CCE Adam" width="600">
</p>

- [Matriz (CSV)](outputs/mlp_relu_relu__categorical_crossentropy__adam_confusion_matrix.csv)
- [History (JSON)](outputs/mlp_relu_relu__categororical_crossentropy__adam_history.json)
- [QC (MD)](outputs/mlp_relu_relu__categorical_crossentropy__adam_history_qc.md)
- [Test report (JSON)](outputs/mlp_relu_relu__categorical_crossentropy__adam_test_report.json)

**SGD**
<p>
  <img src="outputs/mlp_relu_relu__categorical_crossentropy__sgd_curves.png" alt="relu_relu CCE SGD" width="600">
</p>

- [Matriz (CSV)](outputs/mlp_relu_relu__categorical_crossentropy__sgd_confusion_matrix.csv)
- [History (JSON)](outputs/mlp_relu_relu__categorical_crossentropy__sgd_history.json)
- [QC (MD)](outputs/mlp_relu_relu__categorical_crossentropy__sgd_history_qc.md)
- [Test report (JSON)](outputs/mlp_relu_relu__categorical_crossentropy__sgd_test_report.json)

---

### (ReLU, ReLU) — `mse`

**Adam** (⭐ mejor test)
<p>
  <img src="outputs/mlp_relu_relu__mse__adam_curves.png" alt="relu_relu MSE Adam" width="600">
</p>

- [Matriz (CSV)](outputs/mlp_relu_relu__mse__adam_confusion_matrix.csv)
- [History (JSON)](outputs/mlp_relu_relu__mse__adam_history.json)
- [QC (MD)](outputs/mlp_relu_relu__mse__adam_history_qc.md)
- [Test report (JSON)](outputs/mlp_relu_relu__mse__adam_test_report.json)

**SGD**
<p>
  <img src="outputs/mlp_relu_relu__mse__sgd_curves.png" alt="relu_relu MSE SGD" width="600">
</p>

- [Matriz (CSV)](outputs/mlp_relu_relu__mse__sgd_confusion_matrix.csv)
- [History (JSON)](outputs/mlp_relu_relu__mse__sgd_history.json)
- [QC (MD)](outputs/mlp_relu_relu__mse__sgd_history_qc.md)
- [Test report (JSON)](outputs/mlp_relu_relu__mse__sgd_test_report.json)

---

### (ReLU, Tanh) — `categorical_crossentropy`

**Adam**
<p>
  <img src="outputs/mlp_relu_tanh__categorical_crossentropy__adam_curves.png" alt="relu_tanh CCE Adam" width="600">
</p>

- [Matriz (CSV)](outputs/mlp_relu_tanh__categorical_crossentropy__adam_confusion_matrix.csv)
- [History (JSON)](outputs/mlp_relu_tanh__categorical_crossentropy__adam_history.json)
- [QC (MD)](outputs/mlp_relu_tanh__categorical_crossentropy__adam_history_qc.md)
- [Test report (JSON)](outputs/mlp_relu_tanh__categorical_crossentropy__adam_test_report.json)

**SGD**
<p>
  <img src="outputs/mlp_relu_tanh__categorical_crossentropy__sgd_curves.png" alt="relu_tanh CCE SGD" width="600">
</p>

- [Matriz (CSV)](outputs/mlp_relu_tanh__categorical_crossentropy__sgd_confusion_matrix.csv)
- [History (JSON)](outputs/mlp_relu_tanh__categorical_crossentropy__sgd_history.json)
- [QC (MD)](outputs/mlp_relu_tanh__categorical_crossentropy__sgd_history_qc.md)
- [Test report (JSON)](outputs/mlp_relu_tanh__categorical_crossentropy__sgd_test_report.json)

---

### (ReLU, Tanh) — `mse`

**Adam**
<p>
  <img src="outputs/mlp_relu_tanh__mse__adam_curves.png" alt="relu_tanh MSE Adam" width="600">
</p>

- [Matriz (CSV)](outputs/mlp_relu_tanh__mse__adam_confusion_matrix.csv)
- [History (JSON)](outputs/mlp_relu_tanh__mse__adam_history.json)
- [QC (MD)](outputs/mlp_relu_tanh__mse__adam_history_qc.md)
- [Test report (JSON)](outputs/mlp_relu_tanh__mse__adam_test_report.json)

**SGD**
<p>
  <img src="outputs/mlp_relu_tanh__mse__sgd_curves.png" alt="relu_tanh MSE SGD" width="600">
</p>

- [Matriz (CSV)](outputs/mlp_relu_tanh__mse__sgd_confusion_matrix.csv)
- [History (JSON)](outputs/mlp_relu_tanh__mse__sgd_history.json)
- [QC (MD)](outputs/mlp_relu_tanh__mse__sgd_history_qc.md)
- [Test report (JSON)](outputs/mlp_relu_tanh__mse__sgd_test_report.json)



---
## 🧠 Reflexión (síntesis)

- **CCE vs MSE**: Aunque teóricamente **cross-entropy** suele ser mejor para multiclase, en este experimento **MSE+Adam** obtuvo la mejor **test_acc** (0.8781), muy cerca del resto (diferencias de ~0.2–1 pp).  
- **Adam vs SGD**: **Adam** suele **converger más rápido** en pocas épocas; **SGD+momentum** puede alcanzarlo ajustando **LR** y/o con **más épocas**.  
- **Activaciones**: **ReLU** en la primera capa favorece gradientes estables; **Tanh** en la segunda puede suavizar y actuar como regularizador ligero.  
- **Siguientes pasos**: probar **EarlyStopping(restore_best_weights)**, **12–15 épocas**, **Dropout(0.2)** o **L2** suave, y un pequeño **sweep de LR** para SGD.

---



