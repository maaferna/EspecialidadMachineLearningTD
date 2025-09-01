# src/utils/io.py
from __future__ import annotations
import os
from typing import List


def ensure_dir(path: str) -> None:
    """Crea el directorio si no existe (idempotente)."""
    os.makedirs(path, exist_ok=True)


def load_corpus(path: str | None) -> List[str]:
    """
    Carga un corpus desde:
      - Un .txt con 1 documento por línea, o
      - Un directorio con varios .txt
    Si path es None o no existe, devuelve un corpus de ejemplo (ES).
    """
    if path and os.path.exists(path):
        if os.path.isdir(path):
            docs: List[str] = []
            for fname in sorted(os.listdir(path)):
                if fname.lower().endswith(".txt"):
                    with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                        docs.append(f.read().strip())
            return [d for d in docs if d]
        else:
            with open(path, "r", encoding="utf-8") as f:
                docs = [ln.strip() for ln in f if ln.strip()]
            return docs

    # Corpus de ejemplo (ES)
    return [
        "Paciente masculino de 45 años con fiebre leve y congestión nasal. Se sospecha infección viral.",
        "Paciente femenina de 32 años con dolor abdominal persistente, sin fiebre, antecedente de gastritis.",
        "Niño de 8 años con tos seca nocturna y sibilancias. Posible cuadro asmático.",
        "Paciente con cefalea intensa, fotofobia y náuseas. Evaluar migraña.",
        "Adulto mayor con fatiga, palpitaciones y disnea de esfuerzo. Sospecha de anemia.",
        "Paciente postquirúrgico con dolor localizado y leve febrícula. Control de herida y signos de infección.",
        "Erupción cutánea pruriginosa tras nuevo antibiótico. Evaluar reacción alérgica.",
        "Dolor torácico opresivo y diaforesis. Derivar a urgencias por posible síndrome coronario.",
        "Dolor lumbar mecánico sin irradiación. Indicar analgésicos y kinesioterapia.",
        "Pérdida de olfato y gusto, rinorrea y odinofagia. Considerar etiología viral."
    ]
