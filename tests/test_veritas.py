"""Theorem 1 (Veritas Aeterna): admissible updates preserve the invariant set."""
import numpy as np
import pytest
from hypothesis import given, strategies as st

from veda.veritas import (CONSTANTS, PSI_WEIGHTS, TERMS, VALUE_WEIGHTS, Invariants,
                          admissible_step, check_constant_identities, in_box, in_simplex,
                          project_simplex)
from strategies import seeds


def test_constant_identities():
    for name, r in check_constant_identities().items():
        assert r < 1e-12, name


def test_constants_are_immutable():
    with pytest.raises(TypeError):
        CONSTANTS["pi"] = 3.0
    with pytest.raises(TypeError):
        VALUE_WEIGHTS["C"] = 1.0


def test_value_weights_are_normalized_psi_weights():
    w = np.array([VALUE_WEIGHTS[t] for t in TERMS])
    assert in_simplex(w)
    assert np.allclose(w, np.abs([PSI_WEIGHTS[t] for t in TERMS]) / 1.01)


@given(seeds, st.integers(1, 40))
def test_random_update_sequences_preserve_invariants(seed, length):
    rng = np.random.default_rng(seed)
    n, m = 5, 5
    s = Invariants(tuple(rng.dirichlet(np.ones(n))), tuple(rng.random(m)))
    assert s.holds()
    for _ in range(length):
        k = rng.integers(4)
        if k == 0:
            s = admissible_step(s, "box", x=rng.normal(0.5, 2.0, m))
        elif k == 1:
            s = admissible_step(s, "simplex", v=rng.normal(0, 3, n))
        elif k == 2:
            s = admissible_step(s, "mix", w2=rng.dirichlet(np.ones(n)),
                                x2=rng.random(m), lam=float(rng.random()))
        else:
            s = admissible_step(s, "mw", g=rng.normal(0, 5, n), eta=float(rng.uniform(0, 2)))
        assert s.holds()


@given(seeds)
def test_simplex_projection_is_idempotent_and_optimal(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 2, 6)
    p = project_simplex(v)
    assert in_simplex(p) and np.allclose(project_simplex(p), p)
    # optimality: no random simplex point is closer to v
    for q in rng.dirichlet(np.ones(6), 50):
        assert np.linalg.norm(v - p) <= np.linalg.norm(v - q) + 1e-12
