"""Tests for tera.robustness — VAR power diagnostics + permutation tests."""
import numpy as np
import pytest

from tera.robustness import (
    compute_var_obs_to_param_ratio,
    granger_permutation_test,
    multiple_comparison_correction,
)


def test_var_ratio_matches_paper():
    # 48 observations, 3 lags -> 45 effective obs / 7 params = 6.4:1.
    r = compute_var_obs_to_param_ratio(48, 3)
    assert r["ratio"] == pytest.approx(6.4, abs=0.1)
    assert r["status"] == "marginal (5:1 - 10:1)"


def test_var_ratio_underpowered():
    r = compute_var_obs_to_param_ratio(20, 5)
    assert r["ratio"] < 5.0
    assert r["status"] == "underpowered (< 5:1)"


def test_permutation_test_returns_expected_keys():
    rng = np.random.RandomState(0)
    patents = {"A": rng.randint(10, 100, 48), "B": rng.randint(10, 100, 48)}
    standards = {"A": rng.randint(0, 5, 48), "B": rng.randint(0, 5, 48)}
    res = granger_permutation_test(patents, standards, n_permutations=50)
    assert "empirical_p_value" in res
    assert 0.0 <= res["empirical_p_value"] <= 1.0
    # Continuity correction guarantees a strictly positive p-value.
    assert res["empirical_p_value"] > 0.0


def test_multiple_comparison_correction():
    corrected = multiple_comparison_correction(
        {"A": 0.01, "B": 0.5, "C": 0.001}, method="bonferroni"
    )
    assert corrected["A"]["corrected_bonferroni"] == pytest.approx(0.03, abs=1e-9)
    assert corrected["C"]["corrected_bonferroni"] == pytest.approx(0.003, abs=1e-9)
