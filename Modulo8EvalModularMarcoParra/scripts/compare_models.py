#!/usr/bin/env python
import json, subprocess, sys
from pathlib import Path

CONFIGS = [
    "configs/config_mnb_tfidf.yaml",
    "configs/config_w2v_logreg.yaml",
    "configs/config_ft_svm.yaml",
    "configs/config_bert_embed_svm.yaml",
]

def run(cfg):
    print(f"\n== Entrenando con {cfg} ==")
    subprocess.run([sys.executable, "scripts/main_train.py", "--config", cfg], check=True)
    met_path = Path("reports/metrics_cls.json")
    m = json.loads(met_path.read_text())
    return {"config": cfg, "representation": m["representation"], "model": m["model"],
            "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}

def main():
    rows = [run(cfg) for cfg in CONFIGS]
    out_json = Path("reports/metrics_compare.json")
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    # CSV rápido
    import csv
    with open("reports/metrics_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config","representation","model","accuracy","f1_macro"])
        w.writeheader(); w.writerows(rows)
    print("\nComparativa guardada en reports/metrics_compare.{json,csv}")

if __name__ == "__main__":
    main()
