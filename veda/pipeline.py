"""Integration: x -> needs -> creature DAG pass -> anchored harmony (Banach)
-> opinions -> DeGroot consensus -> Nash allocation -> recommendation.
Every stage is bounded (paper Thm 10)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .alpha import blended_daily, daily_score, psi
from .mental import CREATURE_DAG, CREATURES, UNITS, message_pass, needs
from .peace import (consensus_weights, degroot, fj_iterate,
                    friedkin_johnsen_fixed_point, influence_matrix, metropolis_matrix, nash_bargain)
from .safety import CRISIS_MESSAGE, crisis_check
from .veritas import TERMS


def creature_adjacency(dag=CREATURE_DAG, order=UNITS):
    """Undirected version of the DAG: units that exchange messages listen to each other."""
    idx = {u: i for i, u in enumerate(order)}
    A = np.zeros((len(order), len(order)))
    for u, (_, preds) in dag.items():
        for v in preds:
            A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1
    return A


# Doubly stochastic, so DeGroot consensus weights are uniform (no hub bias).
HARMONY_W = metropolis_matrix(creature_adjacency())
# Lazy equal-split alternative; its consensus weights are proportional to degree.
HARMONY_W_LAZY = influence_matrix(creature_adjacency(), self_weight=0.5)


@dataclass
class PipelineResult:
    crisis: bool
    message: str = ""
    psi: float | None = None
    daily: float | None = None
    blended_daily: float | None = None
    needs: dict = field(default_factory=dict)
    activations: dict = field(default_factory=dict)
    harmonized: dict = field(default_factory=dict)
    fj_residuals: list = field(default_factory=list)
    consensus: dict = field(default_factory=dict)
    degroot_spreads: list = field(default_factory=list)
    allocation: dict = field(default_factory=dict)
    recommendation: str = ""


def opinions(z, rho: float):
    """Row u: unit u's proposed attention split = (1-rho) e_u + rho z/sum z."""
    n = len(z)
    share = z / z.sum() if z.sum() > 0 else np.full(n, 1.0 / n)
    return (1 - rho) * np.eye(n) + rho * np.tile(share, (n, 1))


def run_pipeline(x, text: str | None = None, mu: float = 0.5, rho: float = 0.6,
                 floor: float = 0.04, theta=(0.0, 0.0, 0.0), degroot_steps: int = 60):
    if crisis_check(text):
        return PipelineResult(crisis=True, message=CRISIS_MESSAGE)
    if isinstance(x, dict):
        x = [x[t] for t in TERMS]
    n = needs(x)
    tr = message_pass(n)
    h = np.array([tr.h[u] for u in UNITS])
    z, hist = fj_iterate(HARMONY_W, h, mu, steps=60)
    zstar = friedkin_johnsen_fixed_point(HARMONY_W, h, mu)
    O = opinions(z, rho)
    P, spreads = degroot(HARMONY_W, O, steps=degroot_steps)
    pi = consensus_weights(HARMONY_W)
    beta = pi @ O
    alloc = nash_bargain(beta, np.full(len(UNITS), floor))
    best = UNITS[int(np.argmax(alloc))]
    c = CREATURES[best]
    return PipelineResult(
        crisis=False,
        psi=psi(x), daily=daily_score(x), blended_daily=blended_daily(x, theta),
        needs=n, activations=dict(tr.h),
        harmonized=dict(zip(UNITS, z.tolist())),
        fj_residuals=[float(np.max(np.abs(v - zstar))) for v in hist],
        consensus=dict(zip(UNITS, beta.tolist())), degroot_spreads=spreads,
        allocation=dict(zip(UNITS, alloc.tolist())),
        recommendation=f"{best} ({c.virtue}): try {c.skill}",
        message="Reflection aid only, not therapy or diagnosis. Crisis (US): call or text 988.",
    )
