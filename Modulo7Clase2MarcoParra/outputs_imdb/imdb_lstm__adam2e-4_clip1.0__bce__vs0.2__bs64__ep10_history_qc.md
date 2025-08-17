# Reporte de entrenamiento (QC por history)

- **Accuracy final (train)**: 0.9722
- **Accuracy final (val)**: 0.8830
- **Mejor val_accuracy**: 0.8882 (época 5)
- **Gap train-val**: 0.0892
- **Subidas de val_loss** (oscilación): 5
- **Test accuracy**: 0.8638  (rango esperado 0.86–0.89)
- **Cumple rango esperado**: Sí

## Observaciones
- ✔ La accuracy de entrenamiento mejora a lo largo de las épocas.
- ⚠ Posible sobreajuste: gap train-val = 0.089 (> 0.05).
- ⚠ Val_loss presenta oscilaciones frecuentes (5 subidas).
- ✔ Accuracy de test en el rango esperado.