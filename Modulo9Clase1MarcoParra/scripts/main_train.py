#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

import sys
from pathlib import Path

# Añade el root del proyecto al sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.io import load_yaml, save_json, ensure_dir
from src.data.datasets import load_dataset
from src.features.text import build_vectorizer
from src.models.train import split_xy, get_model, evaluate
from src.evaluation.plots import plot_confusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_default.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    df, text_col, label_col = load_dataset(cfg)
    X_train, X_test, y_train, y_test = split_xy(
        df, text_col, label_col,
        test_size=cfg["training"]["test_size"],
        stratify=cfg["training"]["stratify"],
        seed=cfg["project"]["seed"]
    )

    vec = build_vectorizer(cfg)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = get_model(cfg)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)

    metrics = evaluate(y_test, y_pred, labels=("neg","pos"))
    ensure_dir(cfg["project"]["output_dir"])
    save_json(metrics, Path(cfg["project"]["output_dir"]) / "metrics_cls.json")
    plot_confusion(metrics["confusion"], ["neg","pos"], Path(cfg["project"]["figures_dir"]) / "confusion_matrix.png")

    # Guardados simples (pickles opcionales)
    from joblib import dump
    ensure_dir("models")
    dump(vec, "models/vectorizer.joblib")
    dump(clf, "models/model.joblib")

    print("OK — entrenamiento listo. Métricas en reports/, artefactos en models/")

if __name__ == "__main__":
    main()
