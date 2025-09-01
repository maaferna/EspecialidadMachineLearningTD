# src/features/vectorize.py

from typing import Tuple, Dict, List
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def make_vectorizers(
    ngram_range=(1,1),
    min_df=2,
    max_df=0.85,
    lowercase=True,
    sublinear_tf=True,
    norm="l2",
):
    bow = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=lowercase
    )
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=lowercase,
        sublinear_tf=sublinear_tf,  # usa 1+log(tf)
        norm=norm                    # normaliza L2 como en el ejemplo
    )
    return bow, tfidf

def vectorize_all(
    docs: List[str],
    ngram_range=(1,1),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True,
    norm="l2",
):
    bow, tfidf = make_vectorizers(
        ngram_range=ngram_range,
        min_df=min_df, max_df=max_df,
        sublinear_tf=sublinear_tf, norm=norm
    )
    X_bow   = bow.fit_transform(docs)
    X_tfidf = tfidf.fit_transform(docs)
    feat_names = tfidf.get_feature_names_out().tolist()
    return X_bow, X_tfidf, feat_names, bow, tfidf

def top_terms_for_doc(
    tfidf_matrix, feature_names: List[str], doc_idx: int, k: int = 10
) -> List[Tuple[str, float]]:
    row = tfidf_matrix[doc_idx].toarray().ravel()
    idx = np.argsort(-row)[:k]
    return [(feature_names[i], float(row[i])) for i in idx if row[i] > 0]
