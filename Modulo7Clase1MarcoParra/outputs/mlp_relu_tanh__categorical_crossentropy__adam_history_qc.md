# Reporte de entrenamiento (QC por history)

- **Accuracy final (train)**: 0.9130
- **Accuracy final (val)**: 0.8918
- **Mejor val_accuracy**: 0.8918 (época 10)
- **Gap train-val**: 0.0212
- **Subidas de val_loss** (oscilación): 2
- **Test accuracy**: 0.8830  (rango esperado 0.88–0.92)
- **Cumple rango esperado**: Sí

## Observaciones
- ✔ La accuracy de entrenamiento mejora a lo largo de las épocas.
- ✔ Generalización razonable: gap train-val = 0.021.
- ✔ Val_loss desciende de forma relativamente estable (subidas: 2).
- ✔ Accuracy de test en el rango esperado.