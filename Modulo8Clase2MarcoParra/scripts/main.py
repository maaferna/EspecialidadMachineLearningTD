"""Main de orquestación: lee config y ejecuta el pipeline."""
from __future__ import annotations


import argparse
import logging



#!/usr/bin/env python
# --- prioridad al src del proyecto actual ---
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --------------------------------------------

from src.pipelines.run_pipeline import run

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline NLP clínico: spaCy vs NLTK")
    p.add_argument(
    "--config",
    default="configs/config_default.yaml",
    help="Ruta al YAML de configuración",
    )
    p.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    help="Nivel de log",
    )
    return p.parse_args()




def main() -> None:
    args = parse_args()
    logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run(args.config)




if __name__ == "__main__":
    main()