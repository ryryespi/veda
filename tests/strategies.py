import numpy as np
from hypothesis import strategies as st

unit = st.floats(0.0, 1.0, allow_nan=False)
terms5 = st.lists(unit, min_size=5, max_size=5).map(np.array)
seeds = st.integers(0, 2**32 - 1)


def random_simplex(rng, n):
    return rng.dirichlet(np.ones(n))


def random_strongly_connected_W(rng, n, p_extra=0.3):
    """Row-stochastic, positive diagonal, strongly connected (contains a directed ring)."""
    A = np.zeros((n, n))
    perm = rng.permutation(n)
    for k in range(n):
        A[perm[k], perm[(k + 1) % n]] = 1
    A = np.maximum(A, (rng.random((n, n)) < p_extra).astype(float))
    np.fill_diagonal(A, 1)
    W = A * rng.uniform(0.05, 1.0, (n, n))
    return W / W.sum(axis=1, keepdims=True)


def random_dag(rng, n, p=0.4):
    """Random DAG with convex coefficients: node k may read from nodes < k."""
    names = [f"u{k}" for k in rng.permutation(n)]
    dag = {}
    for k, u in enumerate(names):
        preds = [names[j] for j in range(k) if rng.random() < p]
        raw = rng.uniform(0.05, 1.0, len(preds) + 1)
        raw /= raw.sum()
        dag[u] = (float(raw[0]), {v: float(b) for v, b in zip(preds, raw[1:])})
        s = dag[u][0] + sum(dag[u][1].values())
        dag[u] = (dag[u][0] + (1 - s), dag[u][1])  # exact unit sum
    # shuffle dict insertion order so the algorithm must sort
    keys = list(dag)
    rng.shuffle(keys)
    return {k: dag[k] for k in keys}
