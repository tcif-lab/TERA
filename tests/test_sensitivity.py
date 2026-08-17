"""Tests for tera.sensitivity — penalty + policy-window sensitivity."""
import numpy as np
import pytest

from tera.sensitivity import (
    _pelt_penalty_value,
    policy_window_sensitivity,
)


def test_pelt_penalty_values():
    assert _pelt_penalty_value(48, "bic") == pytest.approx(np.log(48), abs=1e-9)
    assert _pelt_penalty_value(48, "aic") == pytest.approx(2.0, abs=1e-9)
    assert _pelt_penalty_value(48, "mbic", gamma=2.0) == pytest.approx(2 * np.log(48), abs=1e-9)


def test_policy_window_sensitivity_counts():
    changepoints = {"Cat_A": [2015, 2020]}
    policy_years = [2015, 2016, 2020]
    df = policy_window_sensitivity(changepoints, policy_years, windows=[1, 2])
    row = df[df["category"] == "Cat_A"].iloc[0]
    # 2015 matches 2015,2016; 2020 matches 2020 -> 3 alignments within ±1yr.
    assert row["±1yr"] == 3
    assert row["±2yr"] == 3


def test_policy_window_sensitivity_empty():
    df = policy_window_sensitivity({"Cat_A": []}, [2015], windows=[1, 2, 3])
    assert df.iloc[0]["±1yr"] == 0
