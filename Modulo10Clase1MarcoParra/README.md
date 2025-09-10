# Módulo 10 — Sesión 1

## Despliegue básico de un modelo de Machine Learning con Flask (API REST)

Este proyecto implementa una **API REST con Flask** que expone un **modelo de clasificación** (Iris) entrenado con técnicas de Machine Learning. Incluye validación de entradas, manejo de errores y un script de pruebas.

---

## Estructura de carpetas

```
flask_ml_api/
├─ environment.yml
├─ README.md
├─ app.py
├─ src/
│  └─ train_model.py
│  └─ test_api.py
└─ models/
   └─ modelo.pkl   # (generado por train_model)
```

> **Nota:** `modelo.pkl` se genera luego de entrenar el modelo con `python -m src.train_model`.

---

## 1) Crear y activar entorno con Conda

```
conda env create -f environment.yml
conda activate flask-ml-api
```

---

## 2) Entrenar el modelo

Ejecuta el script de entrenamiento para generar el artefacto en `models/modelo.pkl`.

```
python -m src.train_model
```

---

## 3) Ejecutar la API Flask

```
python -m app
```

La API estará disponible en:

```
http://127.0.0.1:5000/
```

---

## 4) Probar la API

Prueba los endpoints con herramientas como **curl** o ejecutando el script de test.

### Con curl (ejemplos de uso):

```
curl -X GET http://127.0.0.1:5000/

curl -s -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'

```

### Con script de pruebas:

```
python -m test_api
```

---

## 5) Generar entrega

Empaqueta todos los archivos requeridos en un zip:

```
cd flask_ml_api
zip -r entrega_flask_ml_api.zip .
```

---

## Notas

* Modelo entrenado: RandomForestClassifier sobre Iris (puede cambiarse en `train_model`).
* Validación estricta: `features` debe ser una lista numérica con la cantidad correcta de atributos.
* Respuesta: JSON con clase predicha y, si corresponde, probabilidades.
