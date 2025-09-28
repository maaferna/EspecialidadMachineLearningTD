from __future__ import annotations


import joblib
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


ARTIFACT_PATH = Path("models/modelo.pkl")


def train_and_save_model(random_state: int = 42) -> None:
    """Función para entrenar y guardar un modelo de clasificación usando RandomForest
    en el dataset Iris.
    Args:
        random_state (int, optional): Semilla para reproducibilidad. Defaults to 42.
    Returns:
        None
    """
    # 1) Carga dataset Iris
    data = load_iris()
    X, y = data.data, data.target


    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
        )


    # 3) Modelo (simple, robusto sin escalado)
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        )
    clf.fit(X_train, y_train)


    # 4) Evaluación rápida en test
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))


    # 5) Guardar artefacto (.pkl pedido en la actividad, usando joblib)
    joblib.dump({
        "model": clf,
        "target_names": data.target_names,
        "feature_names": data.feature_names,
        }, ARTIFACT_PATH)
    print(f"\nModelo guardado en: {ARTIFACT_PATH.resolve()}")




if __name__ == "__main__":
    train_and_save_model()