"""Veritas Aeterna: fixed constants treated as axioms, and invariant-preserving updates.

"Eternal truth" is a philosophical framing. Mathematically this module holds
(1) real mathematical constants with checkable identities, (2) a normalized
value-weight vector on the probability simplex, and (3) admissible update
operators that provably keep the state inside the invariant set
    I = { (w, x) : w in simplex Delta_n, x in [0,1]^m }.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

TERMS = ("C", "S", "M", "R", "B")  # calm, support, skills use, avoidance, believability
TERM_NAMES = MappingProxyType({
    "C": "calm", "S": "support", "M": "skills use",
    "R": "avoidance", "B": "thought believability",
})

# Real mathematical constants (axioms of the framework, immutable).
CONSTANTS = MappingProxyType({
    "pi": math.pi,
    "e": math.e,
    "phi": (1 + math.sqrt(5)) / 2,
    "sqrt2": math.sqrt(2),
})

# MindFrame Psi weights (Espinosa, MindFrame combined paper). Signed.
PSI_WEIGHTS = MappingProxyType({"C": 0.18, "S": 0.14, "M": 0.12, "R": -0.26, "B": -0.31})
PSI_WEIGHTS_CONSERVATIVE = MappingProxyType(
    {"C": 0.198, "S": 0.158, "M": 0.130, "R": -0.247, "B": -0.267})

# Normalized value weights omega = |w| / sum|w|  (a point of the simplex).
_abs = np.array([abs(PSI_WEIGHTS[t]) for t in TERMS])
VALUE_WEIGHTS = MappingProxyType(dict(zip(TERMS, (_abs / _abs.sum()).tolist())))


def check_constant_identities(tol: float = 1e-12) -> dict[str, float]:
    """Return residuals of identities the constants must satisfy."""
    phi, pi, e = CONSTANTS["phi"], CONSTANTS["pi"], CONSTANTS["e"]
    return {
        "phi^2 - phi - 1": abs(phi * phi - phi - 1),
        "|e^(i pi) + 1|": abs(cmath.exp(1j * pi) + 1),
        "ln(e) - 1": abs(math.log(e) - 1),
        "sqrt2^2 - 2": abs(CONSTANTS["sqrt2"] ** 2 - 2),
    }


# ---------------------------------------------------------------- invariants
def in_simplex(w, tol: float = 1e-9) -> bool:
    w = np.asarray(w, dtype=float)
    return bool(np.all(w >= -tol) and abs(w.sum() - 1.0) <= tol)


def in_box(x, tol: float = 1e-9) -> bool:
    x = np.asarray(x, dtype=float)
    return bool(np.all(x >= -tol) and np.all(x <= 1 + tol))


@dataclass(frozen=True)
class Invariants:
    """State (w, x) with w in the simplex and x in the unit box."""
    w: tuple
    x: tuple

    def holds(self) -> bool:
        return in_simplex(self.w) and in_box(self.x)


# ------------------------------------------------------- admissible updates
def project_box(x):
    """U1: clip to [0,1]^m (Euclidean projection onto the box)."""
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def project_simplex(v):
    """U2: Euclidean projection onto the simplex (sort-based algorithm;
    Held, Wolfe & Crowder 1974; Duchi et al. 2008)."""
    v = np.asarray(v, dtype=float)
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, n + 1)
    cond = u - (css - 1) / k > 0
    rho = k[cond][-1]
    theta = (css[rho - 1] - 1) / rho
    return np.maximum(v - theta, 0.0)


def mix(w1, w2, lam: float):
    """U3: convex mixing (1-lam) w1 + lam w2, lam in [0,1]."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0,1]")
    return (1 - lam) * np.asarray(w1, float) + lam * np.asarray(w2, float)


def multiplicative_update(w, g, eta: float):
    """U4: exponentiated-gradient step w_i <- w_i exp(eta g_i) / Z."""
    w = np.asarray(w, float)
    z = np.log(np.maximum(w, 1e-300)) + eta * np.asarray(g, float)
    z -= z.max()
    p = np.exp(z) * (w > 0)
    return p / p.sum()


def admissible_step(state: Invariants, kind: str, **kw) -> Invariants:
    """Apply one admissible update; by Theorem 1 the result satisfies the invariants."""
    w, x = np.asarray(state.w, float), np.asarray(state.x, float)
    if kind == "box":
        x = project_box(kw["x"])
    elif kind == "simplex":
        w = project_simplex(kw["v"])
    elif kind == "mix":
        w = mix(w, kw["w2"], kw["lam"])
        x = mix(x, kw["x2"], kw["lam"]) if "x2" in kw else x
    elif kind == "mw":
        w = multiplicative_update(w, kw["g"], kw["eta"])
    else:
        raise ValueError(kind)
    return Invariants(tuple(w.tolist()), tuple(x.tolist()))
