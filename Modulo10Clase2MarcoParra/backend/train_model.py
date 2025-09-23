# train_model.py
from pathlib import Path
import os, joblib, numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUT = MODEL_DIR / "modelo.pkl"
META = MODEL_DIR / "model_meta.joblib"

def load_data(name: str = "breast_cancer"):
    return load_wine() if name == "wine" else load_breast_cancer()

def main(dataset: str = None, random_state: int = None):
    dataset = dataset or os.getenv("DATASET", "breast_cancer")
    random_state = int(random_state or os.getenv("RANDOM_STATE", "42"))

    data = load_data(dataset)
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=random_state, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)

    acc = accuracy_score(yte, yhat)
    f1 = f1_score(yte, yhat, average="binary" if len(np.unique(y)) == 2 else "macro")

    joblib.dump(clf, OUT)
    joblib.dump({
        "dataset": dataset,
        "n_features": X.shape[1],
        "target_names": list(getattr(data, "target_names", [])),
        "feature_names": list(getattr(data, "feature_names", [])),
        "metrics": {"accuracy": float(acc), "f1": float(f1)}
    }, META)

    print(f"[trainer] Modelo: {OUT}")
    print(f"[trainer] Metadatos: {META}")
    print(f"[trainer] Accuracy: {acc:.3f} | F1: {f1:.3f}")

if __name__ == "__main__":
    main()
