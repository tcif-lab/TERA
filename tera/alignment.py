"""
TERA Alignment — cross-source lexical alignment (Stage 1 signal: BM25).

Scores each technology-category description against each standard document
using Okapi BM25 (k1=1.5, b=0.75), producing the STIM matrix that seeds the
cross-source alignment. Category descriptions are bilingual (English +
Chinese); both are concatenated into a single bag-of-words query, matching
the paper's treatment of bilingual corpora.
"""
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class BM25Scorer:
    """Okapi BM25 scorer with scikit-learn for tokenization / IDF."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = None
        self.doc_matrix = None
        self.doc_lens = None
        self.avg_dl = None
        self.N = 0
        self.idf = None

    def fit(self, documents: List[str]) -> "BM25Scorer":
        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            norm=None, use_idf=True, sublinear_tf=False,
        )
        self.doc_matrix = self.vectorizer.fit_transform(documents)
        self.N = self.doc_matrix.shape[0]
        self.doc_lens = self.doc_matrix.sum(axis=1).A1
        self.avg_dl = float(np.mean(self.doc_lens)) if self.N else 1.0
        df = np.array((self.doc_matrix > 0).sum(axis=0)).flatten()
        self.idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
        return self

    def score(self, query_text: str) -> np.ndarray:
        """Score a single query against all fitted documents."""
        q_vec = self.vectorizer.transform([query_text])
        scores = np.zeros(self.N)
        for idx, tf_q in zip(q_vec.indices, q_vec.data):
            if idx >= len(self.idf):
                continue
            tf_d = self.doc_matrix[:, idx].toarray().flatten()
            num = tf_d * (self.k1 + 1)
            den = tf_d + self.k1 * (1 - self.b + self.b * self.doc_lens / self.avg_dl)
            scores += self.idf[idx] * tf_q * num / (den + 1e-10)
        return scores


def load_standard_texts(standards_dir: str, nrows: int = 100) -> Dict[str, str]:
    """
    Load OCR'd standard documents (one CSV per standard) into {name: text}.

    Reads the first `nrows` rows of each CSV and flattens them into a single
    whitespace-normalized text string.
    """
    texts: Dict[str, str] = {}
    if not os.path.isdir(standards_dir):
        return texts
    for fname in sorted(os.listdir(standards_dir)):
        if fname.endswith(".bak") or not fname.endswith(".csv"):
            continue
        try:
            df = pd.read_csv(
                os.path.join(standards_dir, fname),
                encoding="utf-8-sig", header=None, nrows=nrows,
            )
            text = " ".join(df.astype(str).values.flatten().tolist())
            texts[fname.replace(".csv", "")] = re.sub(r"\s+", " ", text)
        except Exception:
            continue
    return texts


def build_stim_matrix(
    queries: Dict[str, str],
    std_texts: Dict[str, str],
    k1: float = 1.5,
    b: float = 0.75,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build the STIM matrix (n_categories x n_standards) of BM25 scores.

    Parameters
    ----------
    queries : {category_name: bilingual query string}
    std_texts : {standard_name: document text}

    Returns
    -------
    (matrix, category_names, standard_names)
    """
    category_names = list(queries.keys())
    standard_names = list(std_texts.keys())

    scorer = BM25Scorer(k1=k1, b=b).fit([std_texts[n] for n in standard_names])
    matrix = np.zeros((len(category_names), len(standard_names)))
    for i, cat in enumerate(category_names):
        matrix[i, :] = scorer.score(queries[cat])
    return matrix, category_names, standard_names


def stim_to_dataframe(
    matrix: np.ndarray, category_names: List[str], standard_names: List[str]
) -> pd.DataFrame:
    """Wrap a STIM matrix into a labeled DataFrame."""
    return pd.DataFrame(matrix, index=category_names, columns=standard_names)
