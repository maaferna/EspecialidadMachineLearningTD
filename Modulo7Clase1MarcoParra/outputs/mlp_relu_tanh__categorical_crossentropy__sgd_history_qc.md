# Reporte de entrenamiento (QC por history)

- **Accuracy final (train)**: 0.9034
- **Accuracy final (val)**: 0.8812
- **Mejor val_accuracy**: 0.8812 (época 10)
- **Gap train-val**: 0.0223
- **Subidas de val_loss** (oscilación): 2
- **Test accuracy**: 0.8690  (rango esperado 0.88–0.92)
- **Cumple rango esperado**: No

## Observaciones
- ✔ La accuracy de entrenamiento mejora a lo largo de las épocas.
- ✔ Generalización razonable: gap train-val = 0.022.
- ✔ Val_loss desciende de forma relativamente estable (subidas: 2).
- ✘ Accuracy de test fuera del rango esperado.