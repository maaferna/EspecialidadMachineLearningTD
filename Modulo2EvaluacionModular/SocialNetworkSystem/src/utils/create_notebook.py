from pathlib import Path
import nbformat as nbf

# 📍 Go two levels up from this file to reach project root
project_root = Path(__file__).resolve().parents[2]

# 🎯 Notebook output directory and path
notebooks_dir = project_root / "notebooks"
notebooks_dir.mkdir(exist_ok=True)
notebook_path = notebooks_dir / "social_network_analysis.ipynb"

# 📝 Create notebook structure
nb = nbf.v4.new_notebook()
nb.cells = []

# 1. Configuración de path para importar desde src
nb.cells.append(nbf.v4.new_code_cell(
    "# ✅ Configurar path para importar módulos desde src/\n"
    "import sys\n"
    "from pathlib import Path\n"
    "sys.path.append(str(Path().resolve().parent / 'src'))"
))

# 2. Instalación de dependencias
nb.cells.append(nbf.v4.new_code_cell(
    "# ✅ Instalación de dependencias necesarias para ejecutar este notebook\n"
    "%pip install numpy==1.23.5 numba==0.56.4 matplotlib faker tqdm scikit-learn pandas seaborn"
))

# 3. Título y descripción
nb.cells.append(nbf.v4.new_markdown_cell(
    "# 📖 Análisis de una Red Social Simulada\n"
    "\n"
    "Este notebook demuestra la generación y análisis de una red social utilizando Programación Orientada a Objetos,\n"
    "NumPy, Numba y algoritmos eficientes para sugerencias de amistad."
))

# 4. Importaciones
nb.cells.append(nbf.v4.new_code_cell(
    "# 📆 Importar librerías necesarias\n"
    "from utils.exceptions import UsuarioExistenteError, UsuarioNoEncontradoError\n"
    "from utils.data_generator import generar_red_social\n"
    "from models.network import RedSocial\n"
    "from optimizations.optimized_ops import (\n"
    "    calcular_amigos_en_comun,\n"
    "    calcular_amigos_en_comun_numba,\n"
    "    convertir_amigos_a_numpy\n"
    ")\n"
    "from algorithms.bfs import sugerencias_amistad\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "import json\n"
    "from time import time\n"
    "from pathlib import Path"
))

# 5. Generación y prueba
nb.cells.append(nbf.v4.new_code_cell(
    "# 💪 Ejecutar prueba sobre distintos escenarios\n"
    "output_dir = Path('../outputs')\n"
    "output_dir.mkdir(exist_ok=True)\n"
    "\n"
    "resultados = []\n"
    "escenarios = [10_000, 50_000, 100_000]\n"
    "\n"
    "for n_usuarios in escenarios:\n"
    "    print(f'\n\U0001f9ea Ejecutando prueba con {n_usuarios} usuarios')\n"
    "    red_social = RedSocial()\n"
    "    red_dict = generar_red_social(n_usuarios=n_usuarios, max_amigos=100)\n"
    "\n"
    "    for usuario in red_dict:\n"
    "        try:\n"
    "            red_social.agregar_usuario(usuario)\n"
    "        except UsuarioExistenteError:\n"
    "            continue\n"
    "\n"
    "    for usuario, amigos in red_dict.items():\n"
    "        for amigo in amigos:\n"
    "            try:\n"
    "                red_social.conectar_usuarios(usuario, amigo)\n"
    "            except UsuarioNoEncontradoError:\n"
    "                continue\n"
    "\n"
    "    red = red_social.obtener_red()\n"
    "    usuarios = list(red.keys())\n"
    "    user_a, user_b = usuarios[0], usuarios[1]\n"
    "    amigos_a = red[user_a]\n"
    "    amigos_b = red[user_b]\n"
    "    amigos_np = convertir_amigos_a_numpy({user_a: amigos_a, user_b: amigos_b})\n"
    "\n"
    "    tiempos = {'usuarios': n_usuarios, 'conjuntos': [], 'numba': []}\n"
    "\n"
    "    for i in range(3):\n"
    "        print(f'\n\U0001f501 Iteración {i+1} con {n_usuarios} usuarios:')\n"
    "        t0 = time()\n"
    "        calcular_amigos_en_comun(amigos_a, amigos_b)\n"
    "        t1 = time()\n"
    "        tiempos['conjuntos'].append(t1 - t0)\n"
    "        print(f'Amigos en común (conjuntos): {t1 - t0:.6f} segundos')\n"
    "\n"
    "        t2 = time()\n"
    "        calcular_amigos_en_comun_numba(amigos_np[user_a], amigos_np[user_b])\n"
    "        t3 = time()\n"
    "        tiempos['numba'].append(t3 - t2)\n"
    "        print(f'Amigos en común (Numba): {t3 - t2:.6f} segundos')\n"
    "\n"
    "    resultados.append(tiempos)\n"
    "\n"
    "    if n_usuarios == 10_000:\n"
    "        print(f'\n\U0001f50d Sugerencias de amistad para {user_a}:')\n"
    "        sugerencias = sugerencias_amistad(red_social, user_a, max_sugerencias=5)\n"
    "        for i, nombre in enumerate(sugerencias, 1):\n"
    "            print(f'{i}. {nombre}')\n"
    "\n"
    "# Guardar JSON\n"
    "with open(output_dir / 'resultados_tiempos.json', 'w') as f:\n"
    "    json.dump(resultados, f, indent=4)"
))

# 6. Visualización
nb.cells.append(nbf.v4.new_code_cell(
    "# 🌐 Visualizar resultados de tiempos\n"
    "usuarios = [r['usuarios'] for r in resultados]\n"
    "conjuntos_prom = [np.mean(r['conjuntos']) for r in resultados]\n"
    "numba_prom = [np.mean(r['numba']) for r in resultados]\n"
    "\n"
    "plt.figure(figsize=(10, 6))\n"
    "plt.plot(usuarios, conjuntos_prom, label='Set (conjuntos)', marker='o')\n"
    "plt.plot(usuarios, numba_prom, label='Numba', marker='s')\n"
    "plt.title('Comparación de Tiempos - Conjuntos vs Numba')\n"
    "plt.xlabel('Número de Usuarios')\n"
    "plt.ylabel('Tiempo Promedio (segundos)')\n"
    "plt.grid(True)\n"
    "plt.legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(output_dir / 'grafico_comparacion_tiempos.png')\n"
    "plt.show()"
))

# ✅ Guardar notebook
nbf.write(nb, notebook_path)
print(f"\u2705 Notebook guardado en: {notebook_path}")

