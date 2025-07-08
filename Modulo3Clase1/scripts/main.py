from utils import generar_datos, ajustar_modelo, graficar_resultado

x, y = generar_datos()
beta = ajustar_modelo(x, y)
graficar_resultado(x, y, beta)
