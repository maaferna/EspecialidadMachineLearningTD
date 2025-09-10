#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

import sys
from pathlib import Path

# Añade el root del proyecto al sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.io import load_yaml
from src.data.datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_default.yaml")
    parser.add_argument("--export_csv", action="store_true", help="Guarda una copia en data/raw/opiniones.csv")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    df, text_col, label_col = load_dataset(cfg)
    print(df.head())

    if args.export_csv:
        out = Path("data/raw/opiniones.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print("CSV exportado a", out)

if __name__ == "__main__":
    main()
