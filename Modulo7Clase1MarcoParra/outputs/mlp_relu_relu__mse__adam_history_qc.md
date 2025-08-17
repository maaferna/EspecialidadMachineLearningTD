# Reporte de entrenamiento (QC por history)

- **Accuracy final (train)**: 0.9056
- **Accuracy final (val)**: 0.8665
- **Mejor val_accuracy**: 0.8862 (época 9)
- **Gap train-val**: 0.0391
- **Subidas de val_loss** (oscilación): 4
- **Test accuracy**: 0.8579  (rango esperado 0.88–0.92)
- **Cumple rango esperado**: No

## Observaciones
- ✔ La accuracy de entrenamiento mejora a lo largo de las épocas.
- ✔ Generalización razonable: gap train-val = 0.039.
- ⚠ Val_loss presenta oscilaciones frecuentes (4 subidas).
- ✘ Accuracy de test fuera del rango esperado.