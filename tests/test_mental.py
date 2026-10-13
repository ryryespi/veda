"""Theorems 5-6 (Mental Project): termination, boundedness, non-expansiveness."""
import numpy as np
import pytest
from hypothesis import given, strategies as st

from veda.mental import CREATURE_DAG, CREATURES, UNITS, message_pass, needs, topological_order
from strategies import random_dag, seeds, terms5


def test_creatures_exact():
    expected = {
        "Lumowl": ("a lantern-eyed night watcher", "reflection", "B", "Check the Facts"),
        "Brakkor": ("a stone-maned beast that walks toward the storm", "courage", "R", "Opposite Action"),
        "Mistfawn": ("a fog spirit that slows every breath", "calm", "C", "TIPP"),
        "Hivewright": ("a many-armed builder laying one brick a day", "discipline", "M", "one easy skill"),
        "Tidelyn": ("a sea serpent linking far shores", "connection", "S", "reach out"),
    }
    for k, (img, virt, term, skill) in expected.items():
        c = CREATURES[k]
        assert (c.image, c.virtue, c.term, c.skill) == (img, virt, term, skill)


def test_creature_dag_counts():
    n = needs([0.5] * 5)
    tr = message_pass(n)
    assert tr.evaluations == 5
    assert tr.messages == sum(len(p) for _, p in CREATURE_DAG.values()) == 6
    assert set(tr.order) == set(UNITS)


def test_cycle_detected():
    cyc = {"a": (0.5, {"b": 0.5}), "b": (0.5, {"a": 0.5})}
    with pytest.raises(ValueError):
        topological_order(cyc)


@given(seeds, st.integers(1, 30))
def test_thm5_random_dag_terminates_with_exact_counts(seed, n):
    rng = np.random.default_rng(seed)
    dag = random_dag(rng, n)
    nv = {u: float(rng.random()) for u in dag}
    tr = message_pass(nv, dag)
    assert tr.evaluations == n
    assert tr.messages == sum(len(p) for _, p in dag.values())
    pos = {u: i for i, u in enumerate(tr.order)}
    assert all(pos[v] < pos[u] for u, (_, p) in dag.items() for v in p)


@given(seeds, st.integers(1, 30))
def test_thm6_bounded_and_nonexpansive(seed, n):
    rng = np.random.default_rng(seed)
    dag = random_dag(rng, n)
    n1 = {u: float(rng.random()) for u in dag}
    n2 = {u: float(rng.random()) for u in dag}
    h1, h2 = message_pass(n1, dag).h, message_pass(n2, dag).h
    assert all(-1e-12 <= v <= 1 + 1e-12 for v in h1.values())
    lhs = max(abs(h1[u] - h2[u]) for u in dag)
    rhs = max(abs(n1[u] - n2[u]) for u in dag)
    assert lhs <= rhs + 1e-12
    # min/max principle
    assert min(h1.values()) >= min(n1.values()) - 1e-12
    assert max(h1.values()) <= max(n1.values()) + 1e-12


@given(terms5)
def test_creature_needs_and_activations_in_unit_interval(x):
    tr = message_pass(needs(x))
    assert all(0 <= v <= 1 for v in tr.h.values())
