# Reporte de entrenamiento (QC por history)

- **Accuracy final (train)**: 0.9109
- **Accuracy final (val)**: 0.8831
- **Mejor val_accuracy**: 0.8928 (época 9)
- **Gap train-val**: 0.0278
- **Subidas de val_loss** (oscilación): 4
- **Test accuracy**: 0.8754  (rango esperado 0.88–0.92)
- **Cumple rango esperado**: No

## Observaciones
- ✔ La accuracy de entrenamiento mejora a lo largo de las épocas.
- ✔ Generalización razonable: gap train-val = 0.028.
- ⚠ Val_loss presenta oscilaciones frecuentes (4 subidas).
- ✘ Accuracy de test fuera del rango esperado.