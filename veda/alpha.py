"""Project Alpha: blending component equations by simplex weights.

B_theta(x) = sum_i alpha_i(theta) f_i(x),  alpha = softmax(theta / tau).
Proven (paper, Thm 2-4): min_i f_i <= B <= max_i f_i; B is (sum alpha_i L_i)-Lipschitz
in x; |B_theta - B_theta'| <= (spread/(2 tau)) ||theta - theta'||_inf.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .veritas import PSI_WEIGHTS, PSI_WEIGHTS_CONSERVATIVE, TERMS


def softmax(theta, tau: float = 1.0):
    if tau <= 0:
        raise ValueError("tau must be > 0")
    z = np.asarray(theta, float) / tau
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def blend(fs: Sequence[Callable], theta, x, tau: float = 1.0) -> float:
    a = softmax(theta, tau)
    vals = np.array([f(x) for f in fs], dtype=float)
    return float(a @ vals)


# ----------------------------------------------------- MindFrame Psi instance
def _vec(x):
    if isinstance(x, dict):
        x = [x[t] for t in TERMS]
    x = np.asarray(x, float)
    if x.shape != (5,) or np.any(x < 0) or np.any(x > 1):
        raise ValueError("terms C,S,M,R,B must be five numbers in [0,1]")
    return x


def psi(x, weights=PSI_WEIGHTS) -> float:
    x = _vec(x)
    w = np.array([weights[t] for t in TERMS])
    return float(w @ x)


def psi_range(weights=PSI_WEIGHTS):
    w = np.array([weights[t] for t in TERMS])
    return float(w[w < 0].sum()), float(w[w > 0].sum())


def daily_score(x, weights=PSI_WEIGHTS) -> float:
    """Daily Score = 100 (Psi - Psi_min) / (Psi_max - Psi_min). For the
    literature weights this is exactly 100 (Psi + 0.57) / 1.01."""
    lo, hi = psi_range(weights)
    return 100.0 * (psi(x, weights) - lo) / (hi - lo)


def oriented_terms(x):
    """y = (C, S, M, 1-R, 1-B): every term oriented so that higher = better."""
    C, S, M, R, B = _vec(x)
    return np.array([C, S, M, 1 - R, 1 - B])


def daily_score_as_blend(x, weights=PSI_WEIGHTS) -> float:
    """Proposition 1: Daily/100 = sum_i omega_i y_i, omega = |w|/sum|w| in simplex."""
    w = np.array([abs(weights[t]) for t in TERMS])
    omega = w / w.sum()
    return float(100.0 * omega @ oriented_terms(x))


def equal_weight_daily(x) -> float:
    return float(100.0 * oriented_terms(x).mean())


SCORING_EQUATIONS = {
    "literature": lambda x: daily_score(x, PSI_WEIGHTS),
    "conservative": lambda x: daily_score(x, PSI_WEIGHTS_CONSERVATIVE),
    "equal": equal_weight_daily,
}


def blended_daily(x, theta=(0.0, 0.0, 0.0), tau: float = 1.0) -> float:
    """Project Alpha blend of the three Daily Score equations."""
    return blend(list(SCORING_EQUATIONS.values()), theta, x, tau)
