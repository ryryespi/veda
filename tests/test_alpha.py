"""Theorems 2-4 and Proposition 1 (Project Alpha)."""
import numpy as np
from hypothesis import given, strategies as st

from veda.alpha import (SCORING_EQUATIONS, blend, blended_daily, daily_score,
                        daily_score_as_blend, psi, psi_range, softmax)
from veda.veritas import PSI_WEIGHTS, PSI_WEIGHTS_CONSERVATIVE, TERMS
from strategies import seeds, terms5

TAU = st.floats(0.05, 10.0)


def affine_components(rng, k, d):
    C = rng.normal(0, 2, (k, d)); b = rng.normal(0, 2, k)
    fs = [lambda x, c=C[i], bb=b[i]: float(c @ x + bb) for i in range(k)]
    L = np.abs(C).sum(axis=1)  # Lipschitz constants w.r.t. the sup norm
    return fs, L


@given(seeds, st.integers(1, 8), TAU)
def test_softmax_in_simplex(seed, k, tau):
    rng = np.random.default_rng(seed)
    a = softmax(rng.normal(0, 5, k), tau)
    assert np.all(a > 0) and abs(a.sum() - 1) < 1e-12


@given(seeds, st.integers(1, 8), TAU)
def test_thm2_blend_between_min_and_max(seed, k, tau):
    rng = np.random.default_rng(seed)
    fs, _ = affine_components(rng, k, 4)
    x = rng.random(4)
    vals = [f(x) for f in fs]
    y = blend(fs, rng.normal(0, 5, k), x, tau)
    assert min(vals) - 1e-9 <= y <= max(vals) + 1e-9


@given(seeds, st.integers(1, 8), TAU)
def test_thm3_lipschitz_in_x(seed, k, tau):
    rng = np.random.default_rng(seed)
    fs, L = affine_components(rng, k, 4)
    th = rng.normal(0, 3, k)
    x, x2 = rng.random(4), rng.random(4)
    a = softmax(th, tau)
    lhs = abs(blend(fs, th, x, tau) - blend(fs, th, x2, tau))
    assert lhs <= (a @ L) * np.max(np.abs(x - x2)) + 1e-9
    assert a @ L <= L.max() + 1e-12


@given(seeds, st.integers(2, 8), TAU)
def test_thm4_lipschitz_in_theta(seed, k, tau):
    rng = np.random.default_rng(seed)
    fs, _ = affine_components(rng, k, 4)
    x = rng.random(4)
    vals = np.array([f(x) for f in fs])
    th, th2 = rng.normal(0, 3, k), rng.normal(0, 3, k)
    lhs = abs(blend(fs, th, x, tau) - blend(fs, th2, x, tau))
    bound = (vals.max() - vals.min()) / (2 * tau) * np.max(np.abs(th - th2))
    assert lhs <= bound + 1e-9


def test_prop1_psi_range_and_constants():
    lo, hi = psi_range()
    assert np.isclose(lo, -0.57) and np.isclose(hi, 0.44) and np.isclose(hi - lo, 1.01)
    assert np.isclose(daily_score([0, 0, 0, 1, 1]), 0.0)
    assert np.isclose(daily_score([1, 1, 1, 0, 0]), 100.0)
    lo2, hi2 = psi_range(PSI_WEIGHTS_CONSERVATIVE)
    assert np.isclose(lo2, -0.514) and np.isclose(hi2, 0.486)


@given(terms5)
def test_prop1_daily_is_convex_blend(x):
    d = daily_score(x)
    assert np.isclose(d, 100 * (psi(x) + 0.57) / 1.01)
    assert np.isclose(d, daily_score_as_blend(x))
    assert -1e-9 <= d <= 100 + 1e-9


@given(terms5, terms5)
def test_prop1_daily_lipschitz(x, x2):
    w = np.abs([PSI_WEIGHTS[t] for t in TERMS])
    lhs = abs(daily_score(x) - daily_score(x2))
    assert lhs <= 100 / 1.01 * (w @ np.abs(x - x2)) + 1e-9
    assert lhs <= 100 * np.max(np.abs(x - x2)) + 1e-9


@given(terms5, st.lists(st.floats(-20, 20), min_size=3, max_size=3), TAU)
def test_blended_daily_scores_bounded(x, theta, tau):
    vals = [f(x) for f in SCORING_EQUATIONS.values()]
    y = blended_daily(x, theta, tau)
    assert min(vals) - 1e-9 <= y <= max(vals) + 1e-9
    assert -1e-9 <= y <= 100 + 1e-9
