from sklearn.ensemble import RandomForestClassifier

def crear_modelo_random_forest(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
    """Crea un modelo RandomForest con parámetros dados"""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
