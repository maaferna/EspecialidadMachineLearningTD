"""Script auxiliar para descargar/generar el dataset pequeño de notas clínicas."""
#!/usr/bin/env python
# --- prioridad al src del proyecto actual ---
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --------------------------------------------

from src.utils.io import load_yaml, save_csv
from src.data.datasets import get_small_clinical_notes

def main(cfg_path: str = "configs/config_default.yaml") -> None:
    cfg = load_yaml(cfg_path)
    df = get_small_clinical_notes(cfg)
    out_csv = cfg["dataset"]["out_csv"]
    print(f"Dataset listo en: {out_csv} — {len(df)} filas")

if __name__ == "__main__":
    main()
