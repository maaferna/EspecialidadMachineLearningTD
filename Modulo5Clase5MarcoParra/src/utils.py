import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cargar_dataset(path="data/Fish.csv"):
    """
    Carga el dataset Fish Market desde archivo CSV local.

    Returns:
        pd.DataFrame: DataFrame cargado con los datos del mercado de peces.
    """
    print(f"📥 Cargando dataset desde {path}...")
    df = pd.read_csv(path)
    print("✅ Dataset cargado exitosamente.")
    print(f"📊 Tamaño del dataset: {df.shape[0]} filas y {df.shape[1]} columnas.")
    print("🔍 Primeras filas del dataset:")
    print(df.head())
    return df


def preprocesar_datos(df, target_col="Weight"):
    """
    Preprocesa el dataset para regresión.

    Pasos:
        - Elimina filas con valores nulos.
        - Codifica la variable categórica 'Species' con one-hot encoding.
        - Estandariza las características numéricas.
    
    Args:
        df (pd.DataFrame): Dataset original.
        target_col (str): Nombre de la columna objetivo.

    Returns:
        tuple: X (features), y (target), scaler (objeto StandardScaler)
    """
    df = df.dropna()

    # Separar variable objetivo
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # One-hot encoding para la columna categórica 'Species'
    X = pd.get_dummies(X, columns=["Species"], drop_first=True)

    # Escalamiento
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, scaler
