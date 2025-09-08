"""Barras de términos principales por documento."""
from __future__ import annotations


from pathlib import Path
from typing import List, Tuple


import matplotlib.pyplot as plt


from src.utils.io import ensure_dir




def plot_top_terms_for_doc(
    terms_with_scores: List[Tuple[str, float]],
    out_path: str | Path,
    title: str = "Top términos (TF-IDF)",
    ) -> None:
    ensure_dir(Path(out_path).parent)
    terms = [t for t, _ in terms_with_scores]
    scores = [s for _, s in terms_with_scores]
    plt.figure(figsize=(8, 4))
    plt.barh(terms[::-1], scores[::-1])
    plt.xlabel("TF-IDF")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()