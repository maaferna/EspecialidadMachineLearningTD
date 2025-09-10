#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from joblib import load

import sys
from pathlib import Path

# Añade el root del proyecto al sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.io import load_yaml, ensure_dir
from src.data.datasets import load_dataset
from src.explain.lime_shap import explain_with_lime, explain_with_shap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_default.yaml")
    parser.add_argument("--method", choices=["lime","shap","both"], default="both")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    df, text_col, label_col = load_dataset(cfg)

    vec = load("models/vectorizer.joblib")
    clf = load("models/model.joblib")

    # Elegimos muestras de test/simple: aleatorias
    np.random.seed(cfg["project"]["seed"])
    texts = df[text_col].astype(str).sample(n=args.samples, random_state=cfg["project"]["seed"]).tolist()

    out_dir = Path(cfg["project"]["explain_dir"])
    if args.method in ("lime","both"):
        explain_with_lime(vec, clf, texts, out_dir, topk=cfg["explain"]["topk"], class_names=("neg","pos"))
    if args.method in ("shap","both"):
        explain_with_shap(vec, clf, texts, out_dir, topk=cfg["explain"]["topk"])

    print(f"OK — explicaciones guardadas en {out_dir}")

if __name__ == "__main__":
    main()
