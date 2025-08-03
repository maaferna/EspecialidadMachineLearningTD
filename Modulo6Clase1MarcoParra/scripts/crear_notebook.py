# create_notebook.py

import nbformat as nbf
from pathlib import Path

def create_notebook():
    # Paths
    notebook_path = Path("notebooks/project_results.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a new notebook
    nb = nbf.v4.new_notebook()
    nb["cells"] = []

    # Add a title markdown cell
    nb["cells"].append(nbf.v4.new_markdown_cell("# 📊 Resultados del Proyecto Clustering Jerárquico"))

    # Add introduction and objectives (you can edit for your case)
    nb["cells"].append(nbf.v4.new_markdown_cell("""
## 🎯 Introducción

Este proyecto tiene como objetivo aplicar **Clustering Jerárquico Aglomerativo** 
usando métodos **Ward** y **Average** sobre datasets clásicos (*Iris* y *Wine*), 
y comparar los resultados mediante visualizaciones de **PCA** y **t-SNE**.

Se busca:
- Explorar la estructura oculta de los datos.
- Visualizar la formación de clusters usando PCA y t-SNE.
- Comparar dendrogramas con distintos métodos de linkage.
- Analizar diferencias entre los métodos y la consistencia de los clusters obtenidos.
"""))

    # Add environment setup cell
    nb["cells"].append(nbf.v4.new_code_cell("""
# ✅ Configuración del entorno
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Ajustar path al root del proyecto
project_root = Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Entorno configurado. Root del proyecto: {project_root}")
"""))

    # Add cell to run the main pipeline
    nb["cells"].append(nbf.v4.new_code_cell("""
# 🚀 Ejecutar pipeline principal
!python -m scripts.main
"""))

    # Add visualization section
    nb["cells"].append(nbf.v4.new_markdown_cell("## 📊 Visualizaciones de Resultados"))

    nb["cells"].append(nbf.v4.new_code_cell("""
from IPython.display import Image, display
import os

# Rutas a las imágenes generadas
output_dir = project_root / "outputs"
imagenes = [
    "pca_varianza_explicada.png",
    "pca_2d.png",
    "pca_3d.png",
    "tsne_average_2.png",
    "tsne_average_3.png",
    "tsne_ward_2.png",
    "tsne_ward_3.png",
    "dendrograma_average_2.png",
    "dendrograma_average_3.png",
    "dendrograma_ward_2.png",
    "dendrograma_ward_3.png"
]

for img in imagenes:
    path = output_dir / img
    if path.exists():
        print(f"Mostrando: {img}")
        display(Image(filename=str(path)))
    else:
        print(f"⚠️ Imagen no encontrada: {path}")
"""))

    # Add conclusions section
    nb["cells"].append(nbf.v4.new_markdown_cell("""
## 📌 Conclusiones

- El método **Ward** mostró clusters más equilibrados y compactos, 
  mientras que **Average** reflejó una estructura más dispersa.
- Con **PCA (2D y 3D)** se observa una separación clara de grupos principales, 
  aunque t-SNE ofrece mayor nitidez en las fronteras locales.
- **t-SNE** preserva relaciones locales entre los puntos, revelando agrupaciones 
  más evidentes aunque a costa de perder proporciones globales.
- El uso de dendrogramas permite observar jerarquías en la formación de clusters, 
  facilitando la elección del número óptimo de grupos.
"""))

    # Save notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"✅ Notebook creado en: {notebook_path}")

if __name__ == "__main__":
    create_notebook()
