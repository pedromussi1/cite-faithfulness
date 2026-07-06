"""Unit tests for the statistics module (bootstrap CIs, paired test, McNemar)."""

from __future__ import annotations

import pytest

from citeval.stats import bootstrap_ci, mcnemar_exact, paired_diff


def test_bootstrap_ci_constant_has_zero_width():
    ci = bootstrap_ci([0.5] * 20, n_boot=500, seed=1)
    assert ci.point == 0.5
    assert ci.lo == 0.5 and ci.hi == 0.5


def test_bootstrap_ci_brackets_point_and_is_reproducible():
    xs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    a = bootstrap_ci(xs, n_boot=2000, seed=42)
    b = bootstrap_ci(xs, n_boot=2000, seed=42)
    assert a.point == pytest.approx(0.5)
    assert a.lo <= a.point <= a.hi
    assert (a.lo, a.hi) == (b.lo, b.hi)  # deterministic under a fixed seed


def test_bootstrap_ci_single_value():
    ci = bootstrap_ci([0.7], n_boot=100, seed=0)
    assert ci.point == 0.7 and ci.lo == 0.7 and ci.hi == 0.7


def test_paired_diff_detects_clear_difference():
    a = [1.0] * 12
    b = [0.0] * 12
    d = paired_diff(a, b, n_boot=2000, seed=0)
    assert d.diff == pytest.approx(1.0)
    assert d.lo > 0  # CI excludes zero
    assert d.significant
    assert d.p_value < 0.05


def test_paired_diff_no_difference_is_not_significant():
    xs = [0.3, 0.5, 0.7, 0.2, 0.9]
    d = paired_diff(xs, xs, n_boot=1000, seed=0)
    assert d.diff == pytest.approx(0.0)
    assert not d.significant
    assert d.p_value == pytest.approx(1.0)


def test_paired_diff_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_diff([1.0, 2.0], [1.0], n_boot=10, seed=0)


def test_mcnemar_all_discordant_one_way_is_significant():
    # A wins on all 8 discordant pairs (a=1, b=0).
    a = [1] * 8
    b = [0] * 8
    r = mcnemar_exact(a, b)
    assert r.n10 == 8 and r.n01 == 0
    assert r.p_value == pytest.approx(2 * (0.5**8))
    assert r.p_value < 0.05


def test_mcnemar_balanced_discordance_not_significant():
    a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    r = mcnemar_exact(a, b)
    assert r.n10 == 5 and r.n01 == 5
    assert r.p_value == pytest.approx(1.0)


def test_mcnemar_concordant_is_p_one():
    a = [1, 0, 1, 0]
    r = mcnemar_exact(a, a)
    assert r.n01 == 0 and r.n10 == 0
    assert r.p_value == 1.0
