"""Métricas por grupo sensible (auditoría simple de sesgo)."""
from __future__ import annotations
import json
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

def _coerce_same_type(y_true: pd.Series, y_pred: pd.Series):
    # Fuerza ambos a str para evitar "Mix of label input types (string and number)"
    yt = y_true.astype(str)
    yp = y_pred.astype(str)
    return yt, yp

def group_metrics(df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str | None = None) -> dict:
    yt, yp = _coerce_same_type(df[y_true_col], df[y_pred_col])
    out = {
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro")),
        "report": classification_report(yt, yp, output_dict=True),
        "confusion": confusion_matrix(yt, yp).tolist(),
    }
    if group_col and group_col in df.columns:
        out["by_group"] = {}
        for g, gdf in df.groupby(group_col):
            gyt, gyp = _coerce_same_type(gdf[y_true_col], gdf[y_pred_col])
            out["by_group"][str(g)] = {
                "support": int(len(gdf)),
                "accuracy": float(accuracy_score(gyt, gyp)),
                "f1_macro": float(f1_score(gyt, gyp, average="macro")),
                "report": classification_report(gyt, gyp, output_dict=True),
                "confusion": confusion_matrix(gyt, gyp).tolist(),
            }
    return out
