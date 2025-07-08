import numpy as np
import sys
from pathlib import Path

# Agregar src/ al path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from sistemas_lineales import ejecutar_sistemas_lineales
from utils import ejecutar_transformacion_por_sistema  

def main():
    # Ejecutar y obtener los sistemas procesados
    sistemas = ejecutar_sistemas_lineales()

    # Aplicar transformación para cada sistema
    ejecutar_transformacion_por_sistema(sistemas)


if __name__ == "__main__":
    main()
