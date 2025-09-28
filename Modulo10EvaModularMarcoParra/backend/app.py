# app.py
from pathlib import Path
import os, joblib, numpy as np
from flask import Flask, jsonify, request

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
MODEL_PATH = MODEL_DIR / "modelo.pkl"
META_PATH  = MODEL_DIR / "model_meta.joblib"

app = Flask(__name__)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró '{MODEL_PATH}'. Ejecuta el servicio trainer primero.")
model = joblib.load(MODEL_PATH)
meta  = joblib.load(META_PATH) if META_PATH.exists() else {}

N_FEATURES   = int(meta.get("n_features", 0)) or getattr(model, "n_features_in_", None)
FEATURE_NAMES = meta.get("feature_names", [])
TARGET_NAMES  = meta.get("target_names", [])

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "API ML lista",
        "model": type(model).__name__,
        "dataset": meta.get("dataset", "unknown"),
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "metrics": meta.get("metrics", {})
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type debe ser application/json"}), 415
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "JSON inválido o vacío"}), 400

    feats = payload.get("features")
    if not isinstance(feats, list):
        return jsonify({"error": "'features' debe ser una lista numérica"}), 400
    if N_FEATURES is not None and len(feats) != N_FEATURES:
        return jsonify({"error": f"'features' debe tener exactamente {N_FEATURES} valores"}), 400

    try:
        X = np.array(feats, dtype=float).reshape(1, -1)
    except Exception:
        return jsonify({"error": "Los valores de 'features' deben ser numéricos"}), 400

    try:
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
        return jsonify({
            "prediction": int(pred) if hasattr(pred, "item") else pred,
            "probabilities": probs,
            "target_names": TARGET_NAMES
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error en predicción: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
