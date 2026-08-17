"""Tests for tera.alignment — BM25 scoring sanity."""
import numpy as np
import pytest

from tera.alignment import BM25Scorer, build_stim_matrix


def test_bm25_shape_and_nonnegative():
    docs = ["robot safety standard", "welding and machining", "sensor and vision"]
    scorer = BM25Scorer(k1=1.5, b=0.75).fit(docs)
    scores = scorer.score("robot safety")
    assert scores.shape == (3,)
    assert (scores >= 0).all()
    # The matching document should score highest.
    assert np.argmax(scores) == 0


def test_bm25_zero_for_unseen_terms():
    docs = ["robot safety standard"]
    scorer = BM25Scorer().fit(docs)
    scores = scorer.score("zzzzz unknown term")
    assert scores[0] == pytest.approx(0.0, abs=1e-9)


def test_build_stim_matrix_shape():
    queries = {"Cat A": "robot safety", "Cat B": "welding machining"}
    std_texts = {"Std 1": "robot safety standard", "Std 2": "welding process"}
    matrix, cats, stds = build_stim_matrix(queries, std_texts)
    assert matrix.shape == (2, 2)
    assert cats == ["Cat A", "Cat B"]
    assert stds == ["Std 1", "Std 2"]
    assert matrix[0, 0] > matrix[0, 1]  # "robot safety" matches Std 1
