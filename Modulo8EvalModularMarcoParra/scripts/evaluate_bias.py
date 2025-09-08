#!/usr/bin/env python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from src.pipelines.run_fairness_audit import run




def parse_args():
    p = argparse.ArgumentParser(description="Auditoría de fairness (métricas por grupo)")
    p.add_argument("--config", default="configs/config_default.yaml")
    return p.parse_args()




def main():
    args = parse_args()
    m = run(args.config)
    print("Fairness por grupo:", m)




if __name__ == "__main__":
    main()