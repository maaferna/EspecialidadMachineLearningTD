#!/usr/bin/env python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from src.pipelines.run_infer import run




def parse_args():
    p = argparse.ArgumentParser(description="Inferencia clasificador NLP clínico")
    p.add_argument("--config", default="configs/config_default.yaml")
    p.add_argument("--text", nargs="+", required=True, help="Texto(s) a clasificar")
    return p.parse_args()




def main():
    args = parse_args()
    labels = run(args.text, args.config)
    for t, y in zip(args.text, labels):
        print(f"[{y}] {t}")




if __name__ == "__main__":
    main()