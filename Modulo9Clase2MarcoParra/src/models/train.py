from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def split_xy(df, text_col, label_col, test_size=0.2, stratify=True, seed=42):
    y = df[label_col].astype(int).to_numpy()
    X = df[text_col].astype(str).tolist()
    return train_test_split(X, y, test_size=test_size, stratify=y if stratify else None, random_state=seed)

def get_model(cfg):
    name = cfg["model"]["name"].lower()
    if name == "logreg":
        p = cfg["model"]["logreg"]
        return LogisticRegression(
            C=float(p.get("C",1.0)),
            class_weight=p.get("class_weight","balanced"),
            max_iter=int(p.get("max_iter",200)),
            n_jobs=-1
        )
    elif name == "random_forest":
        p = cfg["model"]["random_forest"]
        return RandomForestClassifier(
            n_estimators=int(p.get("n_estimators",300)),
            max_depth=p.get("max_depth", None),
            class_weight=p.get("class_weight","balanced"),
            n_jobs=-1,
            random_state=42
        )
    else:
        raise ValueError(f"Modelo no soportado: {name}")

def evaluate(y_true, y_pred, labels=("neg","pos")):
    cm = confusion_matrix(y_true, y_pred).tolist()
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    return {"confusion": cm, "report": rep, "accuracy": acc, "f1_macro": f1m}
