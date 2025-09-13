from __future__ import annotations
from pathlib import Path
import sys

# Root del proyecto (carpeta padre de /scripts)
ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics.txt"

def main() -> int:
    if not METRICS.exists():
        sys.stderr.write(
            f"[!] No se encontró {METRICS}\n"
            "    Debes entrenar primero para generar las métricas:\n"
            "    python -m src.models.train\n"
        )
        return 1

    print(METRICS.read_text(encoding="utf-8"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
