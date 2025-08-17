# Reporte de entrenamiento (QC por history)

- **Accuracy final (train)**: 0.8666
- **Accuracy final (val)**: 0.8627
- **Mejor val_accuracy**: 0.8627 (época 10)
- **Gap train-val**: 0.0039
- **Subidas de val_loss** (oscilación): 0
- **Test accuracy**: 0.8527  (rango esperado 0.88–0.92)
- **Cumple rango esperado**: No

## Observaciones
- ✔ La accuracy de entrenamiento mejora a lo largo de las épocas.
- ✔ Generalización razonable: gap train-val = 0.004.
- ✔ Val_loss desciende de forma relativamente estable (subidas: 0).
- ✘ Accuracy de test fuera del rango esperado.