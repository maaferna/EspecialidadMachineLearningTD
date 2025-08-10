# src/utils.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from scipy import stats


def cargar_dataset(nombre_dataset="iris", usar_minmax=False, con_split=False, test_size=0.2, random_state=42):
    """
    Carga y preprocesa un dataset (Iris o Wine) aplicando escalado.
    Permite devolver dataset completo o dividido en train/test.

    Args:
        nombre_dataset (str): Nombre del dataset ("iris" o "wine").
        usar_minmax (bool): True usa MinMaxScaler, False usa StandardScaler.
        con_split (bool): True devuelve train/test, False devuelve dataset completo.
        test_size (float): Proporción del conjunto de test.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        tuple:
            Si con_split=True → (X_train, X_test, y_train, y_test)
            Si con_split=False → (X, y)
    """
    print(f"📥 Cargando dataset '{nombre_dataset}'...")

    if nombre_dataset == "iris":
        dataset = load_iris(as_frame=True)
    elif nombre_dataset == "wine":
        dataset = load_wine(as_frame=True)
    else:
        raise ValueError("❌ Dataset no soportado. Usa 'iris' o 'wine'.")

    X = dataset.data
    y = dataset.target

    scaler = MinMaxScaler() if usar_minmax else StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 🔧 Corregir nombres de columnas (convertir todo a str)
    X_scaled.columns = X_scaled.columns.astype(str)

    # Eliminar filas con nulos si existen
    print(f"🔍 Verificando NaN iniciales: {X_scaled.isnull().sum().sum()}")
    nulos_antes = X_scaled.isnull().sum().sum()
    if nulos_antes > 0:
        print(f"⚠️ NaNs detectados: {nulos_antes}")
        X_scaled = X_scaled.dropna()
        y = y.loc[X_scaled.index]
        print(f"⚠️ Filas eliminadas por nulos: {nulos_antes}")

    # Outlier básico opcional - VERSIÓN SIMPLIFICADA
    filas_originales = len(X_scaled)
    
    # Método más simple y robusto para outliers
    try:
        # Calcular percentiles en lugar de desviación estándar para evitar problemas
        Q1 = X_scaled.quantile(0.25)
        Q3 = X_scaled.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Aplicar filtro IQR
        outlier_mask = ((X_scaled >= lower_bound) & (X_scaled <= upper_bound)).all(axis=1)
        X_scaled = X_scaled[outlier_mask].copy()
        y = y.loc[X_scaled.index].copy()
        
    except Exception as e:
        print(f"⚠️ Error en filtro de outliers: {e}, manteniendo datos originales")
    
    # Resetear índices
    X_scaled = X_scaled.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # Convertir nombres de columnas a string
    X_scaled.columns = X_scaled.columns.astype(str)
    
    # Verificación final robusta
    nan_final = X_scaled.isnull().sum().sum()
    print(f"✅ Verificación final: {nan_final} NaNs restantes")
    
    if nan_final > 0:
        print("⚠️ Eliminando NaNs finales...")
        X_scaled = X_scaled.dropna()
        y = y.loc[X_scaled.index]
        X_scaled = X_scaled.reset_index(drop=True)
        y = y.reset_index(drop=True)
        print(f"✅ Después de limpieza final: {len(X_scaled)} filas")
    
    print(f"⚠️ Filas eliminadas por outliers: {filas_originales - len(X_scaled)}")

    if con_split:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Asegurar que las columnas sean strings en ambos conjuntos
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)
        
        print(f"✅ Dataset cargado con split: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_scaled.shape[1]} features")
        return X_train, X_test, y_train, y_test
    else:
        print(f"✅ Dataset cargado completo: {len(X_scaled)} filas, {X_scaled.shape[1]} features")
        return X_scaled, y