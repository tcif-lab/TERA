"""Tests for tera.taxonomy — IPC routing correctness."""
import os

import numpy as np
import pandas as pd
import pytest

from tera.taxonomy import load_ipc_routing, classify_patent, build_patent_time_series

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTING = os.path.join(REPO, "config", "ipc_routing.csv")


@pytest.fixture(scope="module")
def maps():
    return load_ipc_routing(ROUTING)


def test_routing_table_loads(maps):
    subclass, cls = maps
    assert "G05B" in subclass and subclass["G05B"] == "Control & Planning"
    assert "B25J" in subclass and subclass["B25J"] == "Manipulation"
    assert "G05" in cls and cls["G05"] == "Control & Planning"


def test_classify_subclass_match(maps):
    subclass, cls = maps
    assert classify_patent("G05B", subclass, cls) == ["Control & Planning"]


def test_classify_multi_category(maps):
    subclass, cls = maps
    cats = classify_patent("G05B;G06N", subclass, cls)
    assert set(cats) == {"Control & Planning", "AI & Learning"}


def test_classify_class_fallback(maps):
    subclass, cls = maps
    # "G05" is a 3-char class -> Control & Planning via class fallback.
    assert classify_patent("G0599", subclass, cls) == ["Control & Planning"]


def test_classify_unclassified(maps):
    subclass, cls = maps
    assert classify_patent("ZZZ", subclass, cls) == []
    assert classify_patent("", subclass, cls) == []
    assert classify_patent(None, subclass, cls) == []


def test_build_time_series_equal_weight(maps):
    subclass, cls = maps
    df = pd.DataFrame({
        "Earliest Priority Date": ["2020-01-01", "2020-06-01"],
        "IPC": ["G05B;G06N", "G05B"],
    })
    ts, n_uncl = build_patent_time_series(df, subclass, cls, [2020])
    # Patent 1 splits 0.5/0.5 across two cats; patent 2 full weight on Control.
    assert ts["Control & Planning"][0] == pytest.approx(1.5)
    assert ts["AI & Learning"][0] == pytest.approx(0.5)
    assert n_uncl == 0
