from pathlib import Path

# Crear estructura base para proyecto "SocialNetworkSystem"
project_root = Path("SocialNetworkSystem")
subdirs = [
    "src",
    "src/models",
    "src/utils",
    "src/algorithms",
    "src/optimizations",
    "data",
    "tests",
    "notebooks",
    "docs",
    "outputs",
    "assets"
]

files = [
    ("README.md", "# 📱 Sistema Integrado de Gestión y Recomendación en una Red Social\n\nDescripción del sistema, flujo, análisis de algoritmos y optimizaciones."),
    ("requirements.txt", "numpy\nnumba\nmatplotlib\nnetworkx\npandas"),
    ("src/__init__.py", ""),
    ("src/main.py", "# Punto de entrada del sistema"),
    ("src/models/user.py", "# Clase Usuario con POO y principios SOLID"),
    ("src/utils/exceptions.py", "# Definición de excepciones personalizadas"),
    ("src/utils/helpers.py", "# Funciones auxiliares como mostrar menú, validaciones, etc."),
    ("src/algorithms/bfs.py", "# Implementación del algoritmo de búsqueda en anchura"),
    ("src/optimizations/timers.py", "# Context manager para medir tiempos"),
    ("src/optimizations/optimized_ops.py", "# Implementaciones con NumPy y Numba")
]

# Crear carpetas
for d in subdirs:
    Path(project_root / d).mkdir(parents=True, exist_ok=True)

# Crear archivos
for file_path, content in files:
    path = project_root / file_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)

print("✅ Proyecto base generado con estructura modular y archivos iniciales.")
