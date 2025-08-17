# Reporte de entrenamiento (QC por history)

- **Accuracy final (train)**: 0.8643
- **Accuracy final (val)**: 0.8535
- **Mejor val_accuracy**: 0.8573 (época 9)
- **Gap train-val**: 0.0108
- **Subidas de val_loss** (oscilación): 1
- **Test accuracy**: 0.8443  (rango esperado 0.88–0.92)
- **Cumple rango esperado**: No

## Observaciones
- ✔ La accuracy de entrenamiento mejora a lo largo de las épocas.
- ✔ Generalización razonable: gap train-val = 0.011.
- ✔ Val_loss desciende de forma relativamente estable (subidas: 1).
- ✘ Accuracy de test fuera del rango esperado.