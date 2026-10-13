"""Peaceful Division: harmonizing disagreeing units.

* DeGroot (1974) averaging p(t+1) = W p(t) with W row-stochastic, positive
  diagonal, strongly connected graph -> consensus (paper Thm 7).
* Friedkin-Johnsen (1990) anchored averaging z <- mu W z + (1-mu) h, a sup-norm
  contraction with unique fixed point (paper Thm 9, Banach).
* Nash (1950) bargaining with weights (Kalai 1977) over an attention budget:
  unique, Pareto-optimal, closed form (paper Thm 8).
"""
from __future__ import annotations

import numpy as np


def influence_matrix(adj, self_weight: float = 0.5):
    """Row-stochastic W: self weight s on the diagonal, the rest split equally
    among out-neighbours in adj (adj[i][j]=1 means i listens to j)."""
    A = np.asarray(adj, float).copy()
    np.fill_diagonal(A, 0)
    n = A.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        deg = A[i].sum()
        if deg == 0:
            W[i, i] = 1.0
        else:
            W[i] = (1 - self_weight) * A[i] / deg
            W[i, i] = self_weight
    return W


def metropolis_matrix(adj):
    """Metropolis-Hastings weights (Xiao & Boyd 2004) on an undirected graph:
    W_ij = 1/(1+max(d_i,d_j)) for edges, W_ii = 1 - sum_j W_ij > 0.
    W is symmetric, hence doubly stochastic, so the consensus is the plain average."""
    A = np.asarray(adj, float).copy()
    np.fill_diagonal(A, 0)
    if not np.allclose(A, A.T):
        raise ValueError("adjacency must be symmetric")
    deg = A.sum(axis=1)
    n = A.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if A[i, j]:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W


def is_row_stochastic(W, tol=1e-12):
    W = np.asarray(W, float)
    return bool(np.all(W >= -tol) and np.allclose(W.sum(axis=1), 1, atol=tol))


def strongly_connected(W) -> bool:
    A = (np.asarray(W) > 0).astype(int)
    n = A.shape[0]
    R = np.eye(n, dtype=int) | A
    for _ in range(n):
        R = ((R @ R) > 0).astype(int)
    return bool(R.all())


def spread(p):
    """max over opinion coordinates of (max_i p_i - min_i p_i)."""
    p = np.asarray(p, float)
    if p.ndim == 1:
        p = p[:, None]
    return float(np.max(np.ptp(p, axis=0)))


def dobrushin(W):
    """tau(W) = 1 - min_{i,j} sum_k min(W_ik, W_jk)  (ergodicity coefficient)."""
    W = np.asarray(W, float)
    n = W.shape[0]
    m = min(np.minimum(W[i], W[j]).sum() for i in range(n) for j in range(n))
    return 1.0 - m


def degroot(W, p0, steps: int = 200, tol: float = 0.0):
    """Iterate p <- W p. p0 may be (n,) or (n,k) (k-dim opinions per unit).
    Returns (p_final, list of spreads)."""
    p = np.asarray(p0, float)
    if p.ndim == 1:
        p = p[:, None]
    spreads = [spread(p)]
    for _ in range(steps):
        p = W @ p
        spreads.append(spread(p))
        if spreads[-1] <= tol:
            break
    return p, spreads


def consensus_weights(W):
    """Left Perron vector pi (pi^T W = pi^T, sum pi = 1), unique when W is
    irreducible; solved as the linear system [(I - W)^T ; 1^T] pi = [0 ; 1]."""
    W = np.asarray(W, float)
    n = W.shape[0]
    M = np.vstack([(np.eye(n) - W).T, np.ones((1, n))])
    rhs = np.concatenate([np.zeros(n), [1.0]])
    pi, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    return pi


def friedkin_johnsen_fixed_point(W, h, mu: float):
    W = np.asarray(W, float)
    n = W.shape[0]
    return np.linalg.solve(np.eye(n) - mu * W, (1 - mu) * np.asarray(h, float))


def fj_iterate(W, h, mu: float, z0=None, steps: int = 100):
    """Banach iteration z <- mu W z + (1-mu) h. Returns (z, history of iterates)."""
    if not 0 <= mu < 1:
        raise ValueError("mu must be in [0,1)")
    h = np.asarray(h, float)
    z = np.zeros_like(h) if z0 is None else np.asarray(z0, float)
    hist = [z.copy()]
    for _ in range(steps):
        z = mu * (W @ z) + (1 - mu) * h
        hist.append(z.copy())
    return z, hist


def nash_bargain(beta, d):
    """Maximize sum_i beta_i log(x_i - d_i) over the simplex.
    Closed form: x_i = d_i + beta_i (1 - sum d). Requires beta>0, sum beta=1,
    d >= 0, sum d < 1."""
    beta = np.asarray(beta, float)
    d = np.asarray(d, float)
    if np.any(beta <= 0) or abs(beta.sum() - 1) > 1e-9:
        raise ValueError("beta must be a strictly positive simplex vector")
    if np.any(d < 0) or d.sum() >= 1:
        raise ValueError("need d >= 0 and sum(d) < 1")
    return d + beta * (1 - d.sum())


def nash_objective(x, beta, d):
    x, beta, d = (np.asarray(v, float) for v in (x, beta, d))
    if np.any(x <= d):
        return -np.inf
    return float(beta @ np.log(x - d))
