"""Mental Project: creature-represented processing units on a DAG.

The creatures are invented, illustrative characters (a philosophical and
mnemonic framing). Mathematically each is a node u of a directed acyclic
graph with update
    h_u = a_u * n_u + sum_{v in pred(u)} b_{uv} * h_v,
    a_u > 0, b_{uv} >= 0, a_u + sum_v b_{uv} = 1,
where n_u in [0,1] is the unit's "need" read from one MindFrame term.
Proven (paper, Thm 5-6): one pass (Kahn order) terminates after exactly |V|
evaluations and |E| messages; every h_u lies in [0,1]; the map n -> h is
non-expansive in the sup norm.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .veritas import TERMS


@dataclass(frozen=True)
class Creature:
    name: str
    image: str
    virtue: str
    term: str        # MindFrame term it reads
    skill: str       # DBT-style skill it suggests
    need_sign: int   # +1: need = x (R,B);  -1: need = 1 - x (C,S,M)


CREATURES = {
    "Lumowl": Creature("Lumowl", "a lantern-eyed night watcher", "reflection", "B", "Check the Facts", +1),
    "Brakkor": Creature("Brakkor", "a stone-maned beast that walks toward the storm", "courage", "R", "Opposite Action", +1),
    "Mistfawn": Creature("Mistfawn", "a fog spirit that slows every breath", "calm", "C", "TIPP", -1),
    "Hivewright": Creature("Hivewright", "a many-armed builder laying one brick a day", "discipline", "M", "one easy skill", -1),
    "Tidelyn": Creature("Tidelyn", "a sea serpent linking far shores", "connection", "S", "reach out", -1),
}
UNITS = ("Mistfawn", "Tidelyn", "Lumowl", "Brakkor", "Hivewright")  # a topological order

# CREATURE_DAG[u] = (a_u, {v: b_uv}); design choices, not empirical estimates.
CREATURE_DAG = {
    "Mistfawn": (1.0, {}),
    "Tidelyn": (1.0, {}),
    "Lumowl": (0.7, {"Mistfawn": 0.3}),
    "Brakkor": (0.6, {"Mistfawn": 0.15, "Lumowl": 0.15, "Tidelyn": 0.10}),
    "Hivewright": (0.7, {"Brakkor": 0.2, "Tidelyn": 0.1}),
}


def validate_dag(dag) -> None:
    for u, (a, preds) in dag.items():
        if a <= 0 or any(b < 0 for b in preds.values()):
            raise ValueError(f"{u}: need a_u > 0 and b_uv >= 0")
        if abs(a + sum(preds.values()) - 1) > 1e-12:
            raise ValueError(f"{u}: a_u + sum b_uv must equal 1")
        for v in preds:
            if v not in dag:
                raise ValueError(f"{u}: unknown predecessor {v}")


def topological_order(dag):
    """Kahn (1962). Raises ValueError if the graph has a cycle."""
    indeg = {u: len(p) for u, (_, p) in dag.items()}
    succ = {u: [] for u in dag}
    for u, (_, p) in dag.items():
        for v in p:
            succ[v].append(u)
    q = deque(sorted(u for u, d in indeg.items() if d == 0))
    order = []
    while q:
        v = q.popleft()
        order.append(v)
        for u in succ[v]:
            indeg[u] -= 1
            if indeg[u] == 0:
                q.append(u)
    if len(order) != len(dag):
        raise ValueError("graph has a cycle")
    return order


def needs(x) -> dict:
    x = dict(zip(TERMS, np.asarray([x[t] for t in TERMS] if isinstance(x, dict) else x, float)))
    out = {}
    for name, c in CREATURES.items():
        v = float(x[c.term])
        out[name] = v if c.need_sign > 0 else 1.0 - v
    return out


@dataclass
class PassTrace:
    h: dict
    order: list
    evaluations: int
    messages: int


def message_pass(n: dict, dag=CREATURE_DAG) -> PassTrace:
    validate_dag(dag)
    order = topological_order(dag)
    h, evals, msgs = {}, 0, 0
    for u in order:
        a, preds = dag[u]
        val = a * n[u]
        for v, b in preds.items():
            val += b * h[v]   # h[v] exists: v precedes u in topological order
            msgs += 1
        h[u] = val
        evals += 1
    return PassTrace(h, order, evals, msgs)
