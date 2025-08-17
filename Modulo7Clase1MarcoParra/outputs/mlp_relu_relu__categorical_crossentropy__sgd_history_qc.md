# Reporte de entrenamiento (QC por history)

- **Accuracy final (train)**: 0.9005
- **Accuracy final (val)**: 0.8766
- **Mejor val_accuracy**: 0.8831 (época 6)
- **Gap train-val**: 0.0239
- **Subidas de val_loss** (oscilación): 3
- **Test accuracy**: 0.8706  (rango esperado 0.88–0.92)
- **Cumple rango esperado**: No

## Observaciones
- ✔ La accuracy de entrenamiento mejora a lo largo de las épocas.
- ✔ Generalización razonable: gap train-val = 0.024.
- ✔ Val_loss desciende de forma relativamente estable (subidas: 3).
- ✘ Accuracy de test fuera del rango esperado.