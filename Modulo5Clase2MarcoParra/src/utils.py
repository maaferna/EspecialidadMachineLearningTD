import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from statsmodels.datasets.macrodata import load_pandas
from statsmodels.tsa.stattools import adfuller

# src/utils.py

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def cargar_y_preprocesar_california():
    """
    Carga y preprocesa el dataset California Housing para regresión con ElasticNet.
    
    Pasos:
    - Verifica valores nulos
    - Aplica log transform a variables sesgadas (opcional)
    - Estandariza las características
    """
    print("📥 Cargando dataset California Housing...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()

    # Verificar nulos
    if df.isnull().sum().sum() > 0:
        print("⚠️ Datos con valores nulos encontrados, eliminando filas nulas...")
        df = df.dropna()
    else:
        print("✅ No se encontraron valores nulos.")
    
    print(f"📊 Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
    # Análisis estadístico del sesgo (skewness) en la distribución de las variables predictoras
    # skew calcula el coeficiente de asimetría (skewness) para cada columna numérica.
    skewness = df.drop(columns=["MedHouseVal"]).skew() # Excluir target
    skewed_cols = skewness[skewness > 1].index.tolist() # Seleccionar columnas con sesgo fuerte (> 1)
    
    # Aplicar transformación logarítmica a columnas sesgadas
    if skewed_cols:
        print(f"🔧 Aplicando log1p a variables sesgadas: {skewed_cols}")
        df[skewed_cols] = df[skewed_cols].apply(lambda x: np.log1p(x))
    else:
        print("✅ No se detectaron variables fuertemente sesgadas.")

    # Separar features y target
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # Escalar features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y



def cargar_y_preprocesar_ingresos():
    """
    Carga y preprocesa el dataset Adult Income para regresión cuantílica.
    Transforma el ingreso a valores numéricos aproximados para predecir.
    Utiliza una variable ordinal sintética como proxy del ingreso.
    Esta variable es binaria (1 = alto, 0 = bajo) y se usa como target.
    El preprocesamiento incluye la eliminación de filas con valores faltantes,
    la conversión de variables categóricas a variables dummy y la estandarización
    de las variables numéricas.
    El objetivo es preparar los datos para un modelo de regresión cuantílica.
    Esta función devuelve un DataFrame con las características transformadas y
    una Serie con la variable objetivo.
    El DataFrame resultante contiene variables numéricas estandarizadas y
    variables categóricas convertidas a variables dummy, lo que permite
    utilizarlo directamente en modelos de regresión cuantílica.
    Además, la variable objetivo es una estimación binaria del ingreso,
    donde 1 indica un ingreso alto (mayor a 50K) y 0 indica un ingreso bajo (menor o igual a 50K).
    Esta transformación es útil para modelos que requieren una variable objetivo numérica,
    como los modelos de regresión cuantílica.
    El preprocesamiento asegura que los datos estén listos para ser utilizados en
    modelos de machine learning, eliminando valores faltantes y normalizando las variables.
    """
    dataset = fetch_openml("adult", version=2, as_frame=True) # Cargar dataset Adult Income
    df = dataset.frame.copy()

    # Detectar nombre real de la columna target
    target_col = 'class'
    if target_col not in df.columns:
        raise ValueError("No se encontró la columna de ingreso en el dataset.")
    print(f"🎯 Columna target detectada: '{target_col}'")

    # Limpiar columnas tipo 'object' (quitar espacios en blanco y asegurar tipo string)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip() 

    # Eliminar filas con valores faltantes
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)
    print(f"✅ Dataset limpio con {df.shape[0]} filas y {df.shape[1]} columnas.")


    # Convertir columna target a variable ordinal 0/1
    df[target_col] = df[target_col].apply(lambda x: 0 if x == "<=50K" else 1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Preprocesamiento de variables categóricas y numéricas
    # Variables categóricas: convertir a variables dummy
    # Variables numéricas: estandarizar
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Pipeline de preprocesamiento
    transformer = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    X_transformed = transformer.fit_transform(X)
    X_transformed = pd.DataFrame(X_transformed)

    print(f"✅ Preprocesamiento completado: X shape = {X_transformed.shape}, y shape = {y.shape}")
    return X_transformed, y


def cargar_y_preprocesar_macro():
    """
    Carga y preprocesa datos macroeconómicos para aplicar VAR:
    - Elimina valores nulos
    - Ajusta el índice temporal
    - Aplica logaritmo y primera diferencia para estacionariedad
    """
    print("📥 Cargando datos macroeconómicos...")
    macro = load_pandas().data

    # Selección de variables relevantes
    df = macro[["realgdp", "realcons", "realinv"]].copy()

    # Establecer índice temporal
    df.index = pd.date_range(start="1959-01-01", periods=len(df), freq="Q")

    # Verificar valores nulos
    if df.isnull().sum().sum() > 0:
        print("⚠️ Datos contienen valores nulos. Se eliminarán...")
        df = df.dropna()

    # Transformación logarítmica para estabilizar varianza
    df_log = df.apply(lambda x: np.log(x))

    # Diferenciación para eliminar tendencia (convertir en estacionario)
    df_diff = df_log.diff().dropna()

    # Verificar estacionariedad con ADF
    # ADF (Augmented Dickey-Fuller) test para cada columna
    '''
    Una serie estacionaria es aquella cuyas propiedades estadísticas (media, varianza, autocorrelación) 
    son constantes en el tiempo.
    '''
    print("📊 Verificando estacionariedad con ADF...")
    for col in df_diff.columns:
        adf_result = adfuller(df_diff[col])
        p_value = adf_result[1]
        print(f"📈 ADF Test '{col}': p-value = {p_value:.4f} {'✅ estacionaria' if p_value < 0.05 else '⚠️ NO estacionaria'}")

    print(f"✅ Macro dataset preprocesado -> shape: {df_diff.shape}")
    return df_diff




