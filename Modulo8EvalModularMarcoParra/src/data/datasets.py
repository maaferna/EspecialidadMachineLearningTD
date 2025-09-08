"""Dataset: carga CSV o genera sintético etiquetado con gravedad y grupo sensible."""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

# Plantillas por clase (textos base); el generador las variará levemente
TEMPLATES: Dict[str, List[str]] = {
    "leve": [
        "Cefalea leve y mareo ocasional. Sin fiebre.",
        "Rinitis alérgica controlada con antihistamínicos.",
        "Molestia gástrica leve postprandial; indica dieta blanda.",
        "Dolor lumbar leve tras caminata prolongada; reposo relativo.",
        "Resfrío común con rinorrea y odinofagia leve.",
        "Hiperglicemia leve en ayuno; plan de control nutricional.",
        "Dermatitis leve en antebrazos; sin signos de infección.",
        "Tensión cefálica intermitente; respuesta a analgésico simple.",
    ],
    "moderado": [
        "Diarrea aguda 24h con malestar general. Hidratación indicada.",
        "Tos seca nocturna; antecedente de asma. Saturación 96%.",
        "Lumbalgia con limitación funcional tras esfuerzo moderado.",
        "Dermatitis con prurito intenso y eritema moderado.",
        "Crisis asmática leve-moderada, uso de inhalador de rescate.",
        "Dolor abdominal tipo cólico, sin signos de irritación peritoneal.",
        "Bronquitis moderada con expectoración mucoide.",
        "Gastritis moderada; pauta con IBP y control.",
    ],
    "severo": [
        "Dolor torácico opresivo irradiado a brazo izquierdo.",
        "Disnea en reposo y edema de tobillos.",
        "Insomnio severo con impacto funcional diurno.",
        "Hemorragia digestiva: melena y mareo al bipedestar.",
        "Saturación 88% al aire ambiente; cianosis peribucal.",
        "Fiebre alta persistente con rigidez de nuca.",
        "Dolor torácico con diaforesis y náuseas; ECG urgente.",
        "Cefalea súbita intensa con vómitos; descartar HSA.",
    ],
}

AUG_SUFFIX = [
    "", " Reevaluar en 24 h.", " Control en 48 h.",
    " Realizar exámenes básicos.", " Reposo relativo.", " Hidratación oral.",
]

def _make_synthetic(n_per_class: int = 20, balance_sensitive: bool = True) -> pd.DataFrame:
    """Genera un dataset sintético balanceado por clase y, opcionalmente, por 'sexo'."""
    rng = np.random.default_rng(42)
    rows = []
    for label, candidates in TEMPLATES.items():
        sexes = ["F", "M"] if balance_sensitive else ["F", "M", "F", "M"]
        for i in range(n_per_class):
            base = rng.choice(candidates)
            suf = rng.choice(AUG_SUFFIX)
            # Variaciones mínimas (p.ej., sin cambiar semántica)
            text = base + suf
            sexo = sexes[i % len(sexes)]
            rows.append({"text": text, "label": label, "sexo": sexo})
    rng.shuffle(rows)
    return pd.DataFrame(rows)

def load_dataset(cfg: dict) -> Tuple[pd.DataFrame, str, str, str | None]:
    """Carga el dataset desde CSV o genera sintético según `cfg["dataset"]`
    
    
    Parameters    ----------
    cfg : dict
        Configuración cargada desde YAML. Debe contener `dataset` con:
          - mode: "csv" o "synthetic"
          - csv_path: ruta al CSV (si mode="csv")
          - text_col: nombre de la columna con el texto
          - label_col: nombre de la columna con la etiqueta de gravedad
          - sensitive_col: (opcional) nombre de la columna con el atributo sensible
          - (si mode="synthetic", puede contener `synthetic` con:)
            - n_per_class: int, cantidad de ejemplos por clase
            - balance_sensitive: bool, si True balancea por grupo sensible (sexo)"""
    ds = cfg["dataset"]
    text_col = ds["text_col"]
    label_col = ds["label_col"]
    sens_col = ds.get("sensitive_col")

    if ds.get("mode") == "csv":
        path = Path(ds["csv_path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"No existe el CSV en {path}. Debe contener columnas: {text_col}, {label_col}, {sens_col or '(opcional)'}")
        df = pd.read_csv(path)
    else:
        syn_cfg = cfg.get("synthetic", {}) or {}
        n_per_class = int(syn_cfg.get("n_per_class", 20))
        balance_sensitive = bool(syn_cfg.get("balance_sensitive", True))
        df = _make_synthetic(n_per_class=n_per_class, balance_sensitive=balance_sensitive)

    # Validaciones mínimas
    for c in [text_col, label_col]:
        if c not in df.columns:
            raise ValueError(f"Columna requerida no encontrada: {c}")
    if sens_col and sens_col not in df.columns:
        sens_col = None  # desactiva fairness si no existe
    return df, text_col, label_col, sens_col
