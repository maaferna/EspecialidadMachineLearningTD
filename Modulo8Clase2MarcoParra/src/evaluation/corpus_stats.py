"""Cálculo de métricas de corpus y comparación entre original y preprocesado."""
from __future__ import annotations


from collections import Counter
from statistics import mean
from typing import Dict, Iterable, List




def doc_length(tokens: List[str]) -> int:
    return len(tokens)




def vocab_size(tokens_list: List[List[str]]) -> int:
    vocab = set()
    for toks in tokens_list:
        vocab.update(toks)
    return len(vocab)




def repetition_ratio(tokens: List[str]) -> float:
    """Proporción de repetición: 1 - (#únicos / #total)."""
    if not tokens:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))




def corpus_summary(tokens_list: List[List[str]]) -> Dict[str, float]:
    lengths = [doc_length(t) for t in tokens_list]
    reps = [repetition_ratio(t) for t in tokens_list]
    return {
        "docs": len(tokens_list),
        "mean_length": float(mean(lengths) if lengths else 0.0),
        "vocab_size": int(vocab_size(tokens_list)),
        "mean_repetition_ratio": float(mean(reps) if reps else 0.0),
        }




def compare_corpora(original: List[List[str]], processed: List[List[str]]) -> Dict[str, dict]:
    return {
    "original": corpus_summary(original),
    "processed": corpus_summary(processed),
    }