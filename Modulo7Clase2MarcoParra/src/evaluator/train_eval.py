import os, json
import numpy as np
from typing import Optional

def compile_model(model, loss: str, optimizer, metrics=None):
    """
    Compila el from typing import Optional
modelo con la función de pérdida, optimizador y métricas especificadas.
    Parámetros:
    - model: Instancia del modelo a compilar.
    - loss: Función de pérdida (str o callable).
    - optimizer: Optimizador (str o instancia de optimizador).
    - metrics: Lista de métricas a evaluar durante el entrenamiento y la evaluación.

    Retorna:
    - model: Modelo compilado.
    Si metrics es None, se usa ['accuracy'] por defecto.
    """
    if metrics is None:
        metrics = ['accuracy']
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics or [])
    return model

def train_model(model, x_train, y_train,
                x_val=None, y_val=None,
                epochs=10, batch_size=32,
                validation_split: Optional[float] = None,
                out_dir="outputs", run_name="run", **fit_kwargs):
    ...
    """
    Entrena el modelo con los datos de entrenamiento y validación.
    Parámetros:
    - model: Modelo a entrenar.
    - x_train: Datos de entrenamiento.
    - y_train: Etiquetas de entrenamiento.
    - x_val: Datos de validación (opcional).
    - y_val: Etiquetas de validación (opcional).
    - epochs: Número de épocas para entrenar.
    - batch_size: Tamaño del batch para el entrenamiento.
    - validation_split: Proporción de datos de entrenamiento para usar como validación.
    - out_dir: Directorio donde guardar los resultados.
    - run_name: Nombre del run para guardar los archivos.
    Si validation_split no es None, se ignoran x_val/y_val y se usa split interno.
    """
    os.makedirs(out_dir, exist_ok=True)
    fit_kwargs = {
        "epochs": epochs,
        "batch_size": batch_size,
        "verbose": 2
    }
    if validation_split is not None:
        fit_kwargs["validation_split"] = validation_split
        history = model.fit(x_train, y_train, **fit_kwargs)
    else:
        history = model.fit(x_train, y_train,
                            validation_data=(x_val, y_val), **fit_kwargs)

    # Guardar history en JSON
    hist_json = os.path.join(out_dir, f"{run_name}_history.json")
    with open(hist_json, "w") as f:
        json.dump(_history_to_dict(history), f, indent=2)
    model.save(os.path.join(out_dir, f"{run_name}_model.keras"))
    return history

def _confusion_matrix(y_true, y_pred, num_classes=10):
    """"Calcula la matriz de confusión para las predicciones.
    Parámetros:
    - y_true: Etiquetas verdaderas.
    - y_pred: Etiquetas predichas.
    - num_classes: Número de clases en el dataset.

    Retorna:
    - Matriz de confusión como un array 2D.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud.")
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _history_to_dict(history_obj):
    # Keras History -> dict serializable
    return {k: [float(x) for x in v] for k, v in history_obj.history.items()}

def evaluate_on_test(model, x_test, y_test, out_dir="outputs", run_name="run"):
    """Evalúa el modelo en el conjunto de test y guarda los resultados.
    Parámetros:
    - model: Modelo a evaluar.
    - x_test: Datos de prueba.
    - y_test: Etiquetas de prueba.
    - out_dir: Directorio donde guardar los resultados.
    - run_name: Nombre del run para guardar los archivos.

    Retorna:
    - accuracy: Precisión del modelo en el conjunto de test.
    - cm: Matriz de confusión como un array 2D.
    """
    # 1) Etiquetas: si vienen one-hot -> índices; si no, se usan tal cual
    if y_test.ndim > 1 and y_test.shape[1] == 10:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    # 2) Predicción
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # 3) Métricas básicas
    acc = float((y_true == y_pred).mean())
    cm = _confusion_matrix(y_true, y_pred, num_classes=10)

    # 4) Guardado
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, f"{run_name}_confusion_matrix.csv"),
               cm, fmt="%d", delimiter=",")
    with open(os.path.join(out_dir, f"{run_name}_test_report.json"), "w") as f:
        json.dump({"accuracy": acc}, f, indent=2)

    return acc, cm