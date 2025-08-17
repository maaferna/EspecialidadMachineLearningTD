# Reflexión sobre combinaciones (GRID)

- **Mejor combinación (test)**: `mse + adam` con **test_acc = 0.8836`.
- **Activaciones**: (relu, tanh), mejor val_acc=0.8888.

## Observaciones
- Cross-entropy suele superar a MSE en clasificación multiclase (optimiza log-verosimilitud).
- Adam converge rápido en pocas épocas; SGD+momentum puede alcanzarlo con más épocas y ajuste de LR.
- ReLU en la primera capa favorece gradientes estables; Tanh en la segunda suaviza y puede ayudar a generalizar.
- Con más recursos: EarlyStopping(restore_best_weights), 12–15 épocas, Dropout(0.2), L2 suave y sweep de LR.

## Tabla de corridas
- mlp_relu_relu__categorical_crossentropy__adam: val_best_acc=0.8928, test_acc=0.8754
- mlp_relu_relu__categorical_crossentropy__sgd: val_best_acc=0.8831, test_acc=0.8706
- mlp_relu_relu__mse__adam: val_best_acc=0.8862, test_acc=0.8579
- mlp_relu_relu__mse__sgd: val_best_acc=0.8573, test_acc=0.8443
- mlp_relu_tanh__categorical_crossentropy__adam: val_best_acc=0.8918, test_acc=0.8830
- mlp_relu_tanh__categorical_crossentropy__sgd: val_best_acc=0.8812, test_acc=0.8690
- mlp_relu_tanh__mse__adam: val_best_acc=0.8888, test_acc=0.8836
- mlp_relu_tanh__mse__sgd: val_best_acc=0.8627, test_acc=0.8527
