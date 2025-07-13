from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split



def evaluar_modelo(nombre, modelo, X_test, y_test, tiempo, mejores_parametros):
    '''
    Evalúa un modelo de clasificación y muestra métricas clave.

    Parámetros:
    - nombre: Nombre del modelo (para mostrar en los resultados)
    - modelo: Modelo entrenado a evaluar
    - X_test: Datos de prueba
    - y_test: Etiquetas verdaderas de prueba
    - tiempo: Tiempo de entrenamiento del modelo
    - mejores_parametros: Parámetros óptimos del modelo

    Retorna un diccionario con las métricas y el modelo.
    '''
    y_pred = modelo.predict(X_test)

    try:
        y_prob = modelo.predict_proba(X_test)

        if y_prob.shape[1] == 2:
            # Clasificación binaria
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            # Multiclase
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    except:
        auc = 0.0
        y_prob = None

    # Soporta binaria y multiclase
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    print(f"\n📊 Evaluación {nombre}:")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"⏱ Tiempo: {tiempo:.2f} seg")

    return {
        "metodo": nombre,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "modelo": modelo,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "tiempo": tiempo,
        "mejores_parametros": mejores_parametros,
    }


def preprocesar_datos_multiclase(df):
    """
    Preprocesamiento recomendado para clasificación multiclase:
    - Elimina columnas constantes
    - Elimina columnas poco informativas (frecuencia < 1%)
    - Imputa valores
    - Escala datos
    - Codifica target multiclase
    - Divide en train/test estratificado
    """

    # Separar X y y
    X = df.drop(columns=["prognosis"])
    y = df["prognosis"]

    # Eliminar columnas constantes
    X = X.loc[:, X.nunique() > 1]

    # Eliminar columnas con baja frecuencia de activación (ej. síntomas muy raros)
    freq_1 = (X.sum(axis=0) / len(X))
    cols_to_keep = freq_1[freq_1 > 0.01].index
    X = X[cols_to_keep]

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Imputación
    imputer = SimpleImputer(strategy="most_frequent")
    X_imputed = imputer.fit_transform(X)

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, label_encoder



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import time


def evaluar_modelo_cv(nombre_modelo, modelo, X, y, duracion, config, cv_folds=5):
    """
    Evalúa el modelo usando validación cruzada estratificada.
    
    - Calcula F1 macro y AUC ROC macro promedio entre los folds.
    - Para clasificación multiclase aplica one-vs-rest.
    - Devuelve un diccionario con métricas.
    """
    print(f"\n📊 Evaluando con validación cruzada ({cv_folds} folds)...")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    f1_scores = []
    roc_auc_scores = []

    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_val)

        # F1 macro
        f1 = f1_score(y_val, y_pred, average="macro")
        f1_scores.append(f1)

        # AUC macro (One-vs-Rest)
        try:
            y_proba = modelo.predict_proba(X_val)
            y_val_bin = label_binarize(y_val, classes=classes)
            auc = roc_auc_score(y_val_bin, y_proba, average="macro", multi_class="ovr")
            roc_auc_scores.append(auc)
        except Exception as e:
            print("⚠️ AUC ROC no pudo calcularse:", e)
            roc_auc_scores.append(np.nan)

    resultado = {
        "modelo": nombre_modelo,
        "f1_macro_mean": np.mean(f1_scores),
        "f1_macro_std": np.std(f1_scores),
        "roc_auc_macro_mean": np.nanmean(roc_auc_scores),
        "roc_auc_macro_std": np.nanstd(roc_auc_scores),
        "duracion": duracion,
        "config": config,
    }

    print(f"✅ F1 macro CV: {resultado['f1_macro_mean']:.4f} ± {resultado['f1_macro_std']:.4f}")
    print(f"✅ ROC AUC macro CV: {resultado['roc_auc_macro_mean']:.4f} ± {resultado['roc_auc_macro_std']:.4f}")

    return resultado
