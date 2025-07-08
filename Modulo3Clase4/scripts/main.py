import sys
from pathlib import Path

# Agregar src/ al path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from funciones import definir_funcion
from utils import calcular_gradiente, calcular_hessiana, encontrar_punto_critico
from visualizador import graficar_funcion_3d
from clasificador import clasificar_punto_critico

def main():
    g, x, y = definir_funcion()
    grad = calcular_gradiente(g, x, y)
    print("🔍 Gradiente calculado:", grad)
    hess = calcular_hessiana(g, x, y)
    print("🔍 Hessiana calculada:", hess)
    soluciones = encontrar_punto_critico(grad, x, y)
    print("🔍 Puntos críticos encontrados:", soluciones)
    
    if soluciones:
        punto = soluciones[0]  # asumimos una sola solución
        tipo = clasificar_punto_critico(hess, punto)

        print("\n✅ Punto crítico encontrado:", punto)
        print("🔍 Tipo de punto:", tipo)

        graficar_funcion_3d(g, x, y, (punto[x], punto[y]), tipo)
    else:
        print("❌ No se encontraron puntos críticos")


if __name__ == "__main__":
    main()
