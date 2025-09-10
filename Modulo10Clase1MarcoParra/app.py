from __future__ import annotations

import json
from typing import Any, List

from flask import Flask, jsonify, request
import numpy as np
import joblib
from pathlib import Path

# Cargar el modelo al iniciar la app (una vez)
ARTIFACT_PATH = Path("models/modelo.pkl")
if not ARTIFACT_PATH.exists():
    raise FileNotFoundError(
        "No se encontró 'modelo.pkl'. Ejecuta primero 'python train_model.py' para generar el modelo."
    )

artifact: dict[str, Any] = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
target_names = artifact.get("target_names", None)
feature_names = artifact.get("feature_names", None)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def root() -> Any:
    """Ruta raíz para comprobar que la API está funcionando.
    Returns:
        JSON con mensaje de estado y detalles del modelo.
    """
    tn = target_names.tolist() if isinstance(target_names, np.ndarray) else target_names
    fn = feature_names.tolist() if isinstance(feature_names, np.ndarray) else feature_names

    return jsonify({
        "message": "API lista",
        "model": type(model).__name__,
        "target_names": tn,
        "feature_names": fn,
    })


def _validate_features(payload: dict[str, Any]) -> np.ndarray:
    """Valida y convierte el payload de entrada en un array numpy adecuado para predecir.
    Args:
        payload (dict): Diccionario con la clave 'features' que contiene una lista de valores numéricos.
    Raises:
        ValueError: Si falta la clave 'features' o la longitud es incorrecta.
        TypeError: Si 'features' no es una lista o contiene valores no numéricos.
    Returns:
        np.ndarray: Array 2D con la muestra para predecir.
    """
    if "features" not in payload:
        raise ValueError("JSON debe contener la clave 'features'.")

    features = payload["features"]
    if not isinstance(features, list):
        raise TypeError("'features' debe ser una lista de valores numéricos.")

    # Validar numérico y longitud esperada (Iris = 4)
    if len(features) != 4:
        raise ValueError("'features' debe tener exactamente 4 valores para Iris.")

    try:
        arr = np.asarray(features, dtype=float)
    except Exception:
        raise TypeError("Todos los elementos de 'features' deben ser numéricos.")

    return arr.reshape(1, -1)


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    try:
        if not request.is_json:
            return jsonify({"error": "El cuerpo de la solicitud debe ser JSON."}), 415

        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "JSON inválido o vacío."}), 400

        X = _validate_features(payload)
        pred_idx = int(model.predict(X)[0])
        pred_proba = getattr(model, "predict_proba", None)
        proba = None
        if callable(pred_proba):
            p = pred_proba(X)[0]
            proba = [float(v) for v in p]

        pred_class = (
            target_names[pred_idx] if target_names is not None and 0 <= pred_idx < len(target_names) else pred_idx
        )

        response = {"prediction": pred_class}
        if proba is not None:
            response["probabilities"] = proba

        return jsonify(response), 200

    except (ValueError, TypeError) as e:
        # Errores del cliente → 400
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Errores inesperados del servidor → 500
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500


@app.route("/predict-demo", methods=["GET"])
def predict_demo():
    sample = [5.1, 3.5, 1.4, 0.2]
    X = np.asarray(sample).reshape(1, -1)
    pred_idx = int(model.predict(X)[0])
    pred_class = target_names[pred_idx]
    return jsonify({"demo_input": sample, "prediction": pred_class})


if __name__ == "__main__":
    # Para desarrollo local
    app.run(host="127.0.0.1", port=5000, debug=True)