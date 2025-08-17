import os, json
import numpy as np

def confusion_matrix(y_true, y_pred, num_classes: int):
    """
    Calcula la matriz de confusión para y_true (índices) y y_pred (índices).
    num_classes: número total de clases.
    Retorna: matriz de confusión como np.ndarray (num_classes x num_classes).
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def evaluate_on_test_generic(
    model,
    x_test,
    y_test,
    num_classes: int,
    out_dir: str = "outputs",
    run_name: str = "run"
):
    """
    Evalúa modelo de clasificación (N clases; para binario usa N=2).
    Guarda: <run>_confusion_matrix.csv y <run>_test_report.json
    Retorna: (accuracy, cm)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Si y_test viene one-hot -> índices
    if hasattr(y_test, "ndim") and y_test.ndim > 1 and y_test.shape[1] == num_classes:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    y_prob = model.predict(x_test, verbose=0)
    if num_classes == 2:
        y_pred = (y_prob.ravel() >= 0.5).astype(int)
    else:
        y_pred = np.argmax(y_prob, axis=1)

    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    print(f"✅ Test accuracy: {acc:.4f} | Confusion matrix:\n{cm}")
    np.savetxt(os.path.join(out_dir, f"{run_name}_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    with open(os.path.join(out_dir, f"{run_name}_test_report.json"), "w") as f:
        json.dump({"accuracy": acc}, f, indent=2)

    return acc, cm
