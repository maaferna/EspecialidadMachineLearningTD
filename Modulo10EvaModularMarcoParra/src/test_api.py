from __future__ import annotations

import json
import time
import requests

BASE_URL = "http://127.0.0.1:5000"


def pretty(resp: requests.Response) -> None:
    """Imprime la respuesta de manera legible.
    Args:
        resp (requests.Response): Respuesta de requests.
    """
    try:
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    except Exception:
        print(resp.text)


def main() -> None:
    """Función principal para probar la API.
    Realiza peticiones GET y POST a la API y muestra las respuestas."""
    # Comprobar raíz
    r = requests.get(f"{BASE_URL}/")
    print("GET / → status:", r.status_code)
    pretty(r)

    samples = [
        {"features": [5.1, 3.5, 1.4, 0.2]},  # válido
        {"features": [6.0, 2.9, 4.5, 1.5]},  # válido
        {"features": [1, 2, 3]},            # inválido: longitud
    ]

    for i, payload in enumerate(samples, 1):
        print(f"\nPOST /predict #{i}")
        r = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=10,
        )
        print("status:", r.status_code)
        pretty(r)
        time.sleep(0.5)


if __name__ == "__main__":
    main()