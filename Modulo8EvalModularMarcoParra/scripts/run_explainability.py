#!/usr/bin/env python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.pipelines.run_explainability import run


def parse_args():
    p = argparse.ArgumentParser(description="Explicabilidad de predicciones clínicas")
    p.add_argument("--config", default="configs/config_default.yaml")
    p.add_argument("--method", choices=["shap", "lime"], default="shap")
    p.add_argument("--samples", type=int, default=3, help="Número de ejemplos de test a explicar")
    return p.parse_args()


def main():
    args = parse_args()
    run(args.config, args.method, args.samples)


if __name__ == "__main__":
    main()
