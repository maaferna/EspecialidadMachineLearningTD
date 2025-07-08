import numpy as np

def resolver_sistema(A, b):
    """
    Resuelve un sistema lineal Ax = b.

    Si A es cuadrada, usa np.linalg.solve. Si no, usa mínimos cuadrados.

    Parámetros:
    - A (ndarray): matriz de coeficientes
    - b (ndarray): vector independiente

    Retorna:
    - x (ndarray): solución del sistema
    """
    if A.shape[0] == A.shape[1]:
        x = np.linalg.solve(A, b)
    else:
        '''
        Resuelve el sistema sobredeterminado o subdeterminado Ax=b en el sentido de mínimos cuadrados:
        Si A no es cuadrada, usa mínimos cuadrados para encontrar la mejor solución.
        La solución se obtiene minimizando ||Ax - b||², lo que implica resolver
        la ecuación normal AᵀAx = Aᵀb.
        La función np.linalg.lstsq resuelve esto de manera eficiente.
        La solución es única si A tiene rango completo.
        Si A no tiene rango completo, la solución es la que minimiza ||Ax - b||².
        Esto es útil en regresión lineal y otros problemas donde A no es cuadrada.
        La función np.linalg.lstsq devuelve la solución de mínimos cuadrados.
        Si A es rectangular, se usa la pseudoinversa para encontrar la solución óptima.
        np.linalg.lstsq(A, b, rcond=None) devuelve la solución de mínimos cuadrados.
        
        x:          el vector solución, igual que si usaras np.linalg.solve en un sistema compatible
        residuals:  suma de residuos cuadrados (si el sistema está sobre determinado y compatible)
        rank:       rango efectivo de la matriz A
        s:          valores singulares de A (de la descomposición SVD)
        '''


        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        print(f"Shape de x: {x.shape}")
        if residuals.size > 0:
            print(f"Residuos: {residuals}")
        else:
            print("No hay residuos, el sistema es compatible o subdeterminado.")
        print("*"*20)
        print(f"Shape de x: {x.shape}")
        print(f"Solución x: {x}")
        print(f"Shape de A: {A.shape}")
        print(f"Shape de b: {b.shape}")
        print(f"Rango de A: {rank}")
        print(f"Valores singulares de A: {s}")
        print(f"Residuos: {residuals if residuals.size > 0 else 'N/A'}")
        print(f"Forma de A: {A.shape}, Forma de b: {b.shape}, Forma de x: {x.shape}")
    
    print(f"Shape de x: {x.shape}")
    print(f"Solución x: {x}")
    return x
