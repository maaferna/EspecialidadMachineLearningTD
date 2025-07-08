import sys
from pathlib import Path

# Asegura acceso a src/ como módulo
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np

from src.excepciones import (
    MatrizSingularError,
    ParametrosNoConvergentesError,
    DatosInsuficientesError,
)
from src.modelo_analitico import calculo_cerrado_regresion
from src.optimizadores import gradient_descent


# --- Test para MatrizSingularError ---
def test_matriz_singular_error():
    X = np.array([1, 1, 1])
    y = np.array([2, 2, 2])
    with pytest.raises(MatrizSingularError) as exc_info:
        calculo_cerrado_regresion(X, y)
    assert "XᵀX es singular" in str(exc_info.value)

def test_datos_insuficientes_error():
    print("\n🔍 Test: DatosInsuficientesError - Simulando entrada vacía.")
    try:
        raise DatosInsuficientesError()
    except DatosInsuficientesError as e:
        print(f"✅ Excepción capturada: {e}")
        assert str(e) == "No hay suficientes datos para entrenar el modelo."


def test_parametros_no_convergentes_error():
    print("\n🔍 Test: ParametrosNoConvergentesError - Forzando divergencia con lr muy alto.")
    X = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    with pytest.raises(ParametrosNoConvergentesError) as exc_info:
        gradient_descent(X, y, lr=100, n_iter=50)
    print(f"✅ Excepción capturada: {exc_info.value}")
    assert "no converge" in str(exc_info.value)
