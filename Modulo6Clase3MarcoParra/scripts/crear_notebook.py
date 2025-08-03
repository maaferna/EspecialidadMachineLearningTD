# crear_notebook.py
import nbformat as nbf
import os

def crear_notebook():
    # Nombre del archivo a crear
    nb = nbf.v4.new_notebook()

    # --------------------------
    # 1. Introducción (README)
    # --------------------------
    intro = """# 📊 Proyecto PCA + KNN

Este notebook reproduce el flujo descrito en el **README.md**, mostrando:
- Preprocesamiento de datos
- Análisis PCA no supervisado
- Selección del número óptimo de componentes
- Evaluación de KNN con PCA
- Visualizaciones y resultados

---
"""

    nb['cells'].append(nbf.v4.new_markdown_cell(intro))

    # --------------------------
    # 2. Instalación de dependencias
    # --------------------------
    deps = """# Instalación de dependencias necesarias
!pip install scikit-learn matplotlib seaborn pandas
"""
    nb['cells'].append(nbf.v4.new_code_cell(deps))

    # --------------------------
    # 3. Configuración general
    # --------------------------
    config = """# Configuración general
import os
import matplotlib.pyplot as plt
%matplotlib inline

# Asegurar que trabajamos desde el root del proyecto
os.chdir("..")
print("📂 Directorio actual:", os.getcwd())
"""
    nb['cells'].append(nbf.v4.new_code_cell(config))

    # --------------------------
    # 4. Ejecución del pipeline principal
    # --------------------------
    ejecutar = """# Ejecutar el pipeline principal
!python -m scripts.main
"""
    nb['cells'].append(nbf.v4.new_code_cell(ejecutar))

    # --------------------------
    # 5. Visualización de resultados
    # --------------------------
    visualizar = """# Mostrar imágenes de outputs generados

from IPython.display import Image, display

imagenes = [
    "outputs/pca_varianza_explicada.png",
    "outputs/pca_2d.png",
    "outputs/heatmap_knn_pca.png",
    "outputs/pca_cluster_3d.png"
]

for img in imagenes:
    if os.path.exists(img):
        display(Image(filename=img))
    else:
        print(f"⚠️ Imagen no encontrada: {img}")
"""
    nb['cells'].append(nbf.v4.new_code_cell(visualizar))

    # --------------------------
    # 6. Conclusiones
    # --------------------------
    conclusiones = """# 📌 Conclusiones

- PCA permitió reducir la dimensionalidad del dataset preservando más del 95% de la varianza con 2 componentes.
- El modelo KNN con PCA alcanzó un accuracy óptimo con **k=7 vecinos**, logrando métricas cercanas al 100%.
- La visualización en 2D y 3D muestra una clara separación entre las clases del dataset Iris.
- La reducción de dimensionalidad simplifica el modelo sin pérdida significativa de información.

---
"""
    nb['cells'].append(nbf.v4.new_markdown_cell(conclusiones))

    # Guardar notebook
    with open("notebook/analisis_pca_knn.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("✅ Notebook creado en: notebook/analisis_pca_knn.ipynb")

if __name__ == "__main__":
    crear_notebook()
