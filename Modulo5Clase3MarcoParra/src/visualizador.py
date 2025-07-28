import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Modelos soportados
MODELOS = ["RandomForest", "AdaBoost", "XGBoost", "LightGBM", "CatBoost"]

def graficar_accuracy_mejores_modelos(output_dir="outputs"):
    """Genera gráfico de barras comparando la accuracy de los mejores modelos."""
    registros = []
    for modelo in MODELOS:
        path = os.path.join(output_dir, f"{modelo.lower()}_mejor.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            registros.append({"Modelo": modelo, "Accuracy": df["accuracy"].values[0]})
        else:
            print(f"⚠️ No se encontró {modelo.lower()}_mejor.csv")

    if registros:
        df_modelos = pd.DataFrame(registros)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Modelo", y="Accuracy", data=df_modelos, palette="viridis")
        plt.title("Comparación de Accuracy - Mejores Modelos")
        plt.ylim(0.5,1)

        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(output_dir, "comparacion_accuracy.png"))
        plt.close()
    else:
        print("❌ No se encontró ningún archivo *_mejor.csv para graficar.")

def graficar_todos_los_resultados(output_dir="outputs"):
    """Genera un gráfico boxplot con la distribución de accuracy para todas las combinaciones de hiperparámetros."""
    df_total = []

    for modelo in MODELOS:
        path = os.path.join(output_dir, f"{modelo.lower()}_all_results.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Modelo"] = modelo
            df_total.append(df)
        else:
            print(f"⚠️ No se encontró {modelo.lower()}_all_results.csv")

    if df_total:
        df_all = pd.concat(df_total, ignore_index=True)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Modelo", y="mean_test_score", data=df_all, palette="coolwarm")
        plt.title("Distribución de Accuracy - Todas las configuraciones por modelo")
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(output_dir, "accuracy_todos_modelos.png"))
        plt.close()
    else:
        print("❌ No se encontraron resultados completos para ningún modelo.")

def graficar_matriz_confusion(modelo: str, output_dir="outputs"):
    """Genera un heatmap de la matriz de confusión del modelo especificado."""
    archivo = f"matriz_confusion_{modelo.lower()}.csv"
    path = os.path.join(output_dir, archivo)

    if not os.path.exists(path):
        print(f"❌ No se encontró la matriz de confusión para {modelo}")
        return

    df = pd.read_csv(path, index_col=0)
    cm = df.values

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["<=50K", ">50K"],
                yticklabels=["<=50K", ">50K"])
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - {modelo}")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{modelo.lower()}.png"))
    plt.close()

def graficar_todas_las_matrices_confusion(output_dir="outputs"):
    """Genera todas las matrices de confusión en un solo paso."""
    for modelo in MODELOS:
        graficar_matriz_confusion(modelo, output_dir)

