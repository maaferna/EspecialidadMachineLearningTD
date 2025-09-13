from __future__ import annotations
import joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from src.data.load_heart import load_heart_dataframe

IMAGES = Path("images"); IMAGES.mkdir(parents=True, exist_ok=True)
ARTIFACT_PATH = Path("models/model_rf.joblib")

def main(random_state: int = 42, n_examples: int = 3):
    """Genera explicaciones LIME sobre el espacio transformado (sin inverse_transform).
    guarda figuras en images/lime_explanation_case_*.png
    Args:
        random_state (int): semilla para reproducibilidad.
        n_examples (int): número de explicaciones locales a generar.
    Returns:
        None
    """
    df, target = load_heart_dataframe()
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    artifact = joblib.load(ARTIFACT_PATH)
    pipe = artifact["model"]
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # Transformado (denso)
    X_train_enc = pre.fit_transform(X_train)
    X_test_enc  = pre.transform(X_test)
    if hasattr(X_train_enc, "toarray"):
        X_train_enc = X_train_enc.toarray()
    if hasattr(X_test_enc, "toarray"):
        X_test_enc = X_test_enc.toarray()

    feature_names = pre.get_feature_names_out()
    class_names = [str(c) for c in sorted(y.unique())]

    explainer = LimeTabularExplainer(
        training_data=X_train_enc,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        random_state=random_state
    )

    # predict_fn opera directamente sobre el espacio transformado
    predict_fn = lambda data: clf.predict_proba(data)

    idxs = np.random.RandomState(random_state).choice(len(X_test_enc), size=min(n_examples, len(X_test_enc)), replace=False)
    for i, idx in enumerate(idxs, 1):
        x = X_test_enc[idx]
        exp = explainer.explain_instance(
            data_row=x,
            predict_fn=predict_fn,
            num_features=10
        )
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(IMAGES/f"lime_explanation_case_{i}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print("LIME listo: figuras en images/")

if __name__ == "__main__":
    main()
