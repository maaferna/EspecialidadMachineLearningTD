#!/usr/bin/env python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.pipelines.run_train import run

def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento de clasificador clínico")
    p.add_argument("--config", default="configs/config_default.yaml",
                   help="Ruta al YAML de configuración")
    p.add_argument("--model", default=None,
                   help="Override del modelo: multinomial_nb|linear_svm|logreg")
    p.add_argument("--rep",   default=None,
                   help="Override de representación: tfidf|word2vec|fasttext|transformer_embed")
    p.add_argument("--tag",   default="",
                   help="Sufijo opcional para nombrar la corrida (no sobrescribir)")
    return p.parse_args()

def main():
    args = parse_args()
    _ = run(
        cfg_path=args.config,
        override_model=args.model,
        override_rep=args.rep,
        run_tag=args.tag,
    )
    print("OK — métricas guardadas.")

if __name__ == "__main__":
    main()
