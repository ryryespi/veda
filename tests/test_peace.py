"""Theorems 7-9 (Peaceful Division)."""
import numpy as np
from hypothesis import given, strategies as st

from veda.peace import (consensus_weights, degroot, dobrushin, fj_iterate,
                        friedkin_johnsen_fixed_point, influence_matrix, is_row_stochastic,
                        metropolis_matrix, nash_bargain, nash_objective, spread,
                        strongly_connected)
from veda.pipeline import HARMONY_W, HARMONY_W_LAZY, creature_adjacency
from strategies import random_simplex, random_strongly_connected_W, seeds

N = st.integers(2, 12)


@given(seeds, N)
def test_thm7_spread_monotone_and_contracts(seed, n):
    rng = np.random.default_rng(seed)
    W = random_strongly_connected_W(rng, n)
    assert is_row_stochastic(W) and strongly_connected(W) and np.all(np.diag(W) > 0)
    p = rng.random(n) * 10
    _, sp = degroot(W, p, steps=50)
    assert all(b <= a + 1e-12 for a, b in zip(sp, sp[1:]))
    Wk = np.linalg.matrix_power(W, n - 1)
    delta = Wk.min()
    assert delta > 0                              # primitivity (Lemma 7a)
    assert dobrushin(Wk) <= 1 - n * delta + 1e-12  # Lemma 7b
    assert spread(Wk @ p) <= (1 - n * delta) * spread(p) + 1e-12


@given(seeds, N, st.integers(1, 4))
def test_thm7_converges_to_pi_weighted_consensus(seed, n, k):
    rng = np.random.default_rng(seed)
    W = random_strongly_connected_W(rng, n, p_extra=0.5)
    p0 = rng.random((n, k))
    pi = consensus_weights(W)
    assert np.all(pi > 0) and np.isclose(pi.sum(), 1) and np.allclose(pi @ W, pi, atol=1e-10)
    p, sp = degroot(W, p0, steps=5000, tol=1e-13)
    assert sp[-1] <= 1e-10
    assert np.allclose(p, np.tile(pi @ p0, (n, 1)), atol=1e-8)
    # consensus inside the convex hull of initial opinions (coordinatewise)
    assert np.all(p[0] >= p0.min(axis=0) - 1e-12) and np.all(p[0] <= p0.max(axis=0) + 1e-12)


@given(seeds, st.integers(2, 10))
def test_metropolis_doubly_stochastic_gives_average(seed, n):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < 0.4).astype(float)
    for k in range(n - 1):
        A[k, k + 1] = 1  # path -> connected
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    W = metropolis_matrix(A)
    assert is_row_stochastic(W) and np.allclose(W, W.T) and np.all(np.diag(W) > 0)
    assert np.allclose(consensus_weights(W), np.full(n, 1 / n), atol=1e-10)


def test_creature_harmony_matrices():
    A = creature_adjacency()
    assert np.allclose(consensus_weights(HARMONY_W), 0.2)
    deg = A.sum(axis=1)
    assert np.allclose(consensus_weights(HARMONY_W_LAZY), deg / deg.sum())  # hub bias


@given(seeds, st.integers(2, 10))
def test_thm8_nash_unique_pareto(seed, n):
    rng = np.random.default_rng(seed)
    beta = random_simplex(rng, n) * 0.98 + 0.02 / n
    beta /= beta.sum()
    d = random_simplex(rng, n) * rng.uniform(0, 0.9)
    x = nash_bargain(beta, d)
    assert np.isclose(x.sum(), 1) and np.all(x > d)
    best = nash_objective(x, beta, d)
    for _ in range(200):
        y = d + random_simplex(rng, n) * (1 - d.sum())  # feasible, y >= d
        assert nash_objective(y, beta, d) <= best + 1e-12
        # Pareto: y cannot weakly dominate x with a strict gain
        assert not (np.all(y >= x - 1e-15) and np.any(y > x + 1e-9))


@given(seeds, N, st.floats(0.0, 0.99))
def test_thm9_banach_contraction(seed, n, mu):
    rng = np.random.default_rng(seed)
    W = random_strongly_connected_W(rng, n)
    h = rng.random(n)
    T = lambda z: mu * W @ z + (1 - mu) * h
    z1, z2 = rng.normal(0, 3, n), rng.normal(0, 3, n)
    assert np.max(np.abs(T(z1) - T(z2))) <= mu * np.max(np.abs(z1 - z2)) + 1e-12
    zs = friedkin_johnsen_fixed_point(W, h, mu)
    assert np.allclose(T(zs), zs, atol=1e-12)
    assert np.all(zs >= h.min() - 1e-12) and np.all(zs <= h.max() + 1e-12)
    z0 = rng.random(n)
    _, hist = fj_iterate(W, h, mu, z0=z0, steps=40)
    e0 = np.max(np.abs(z0 - zs))
    for k, zk in enumerate(hist):
        assert np.max(np.abs(zk - zs)) <= mu ** k * e0 + 1e-11
    for k in range(1, len(hist)):  # a-posteriori bound
        assert np.max(np.abs(hist[k] - zs)) <= mu / (1 - mu) * np.max(np.abs(hist[k] - hist[k - 1])) + 1e-11
    h2 = rng.random(n)
    zs2 = friedkin_johnsen_fixed_point(W, h2, mu)
    assert np.max(np.abs(zs - zs2)) <= np.max(np.abs(h - h2)) + 1e-12
