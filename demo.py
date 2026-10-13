"""Project Veda demo. Example inputs below are made-up illustration values,
not data from any person. Writes paper/results.json and plot data files.

Veda is a reflection aid, not therapy. Crisis (US): call or text 988.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from veda import run_pipeline, CREATURES                     # noqa: E402
from veda.alpha import SCORING_EQUATIONS, softmax              # noqa: E402
from veda.mental import UNITS                                  # noqa: E402
from veda.peace import (consensus_weights, degroot, dobrushin, # noqa: E402
                        friedkin_johnsen_fixed_point)
from veda.pipeline import HARMONY_W, HARMONY_W_LAZY, opinions   # noqa: E402
from veda.veritas import check_constant_identities              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper")
os.makedirs(OUT, exist_ok=True)

EXAMPLE = {"C": 0.4, "S": 0.6, "M": 0.3, "R": 0.7, "B": 0.8}  # illustration only


def main():
    x = [EXAMPLE[t] for t in "CSMRB"]
    r = run_pipeline(x)
    print("Project Veda demo (example values, not real data)")
    print("  input C,S,M,R,B =", x)
    print(f"  Psi = {r.psi:.4f}   Daily Score = {r.daily:.2f}   blended Daily = {r.blended_daily:.2f}")
    print("  equations:", {k: round(f(x), 2) for k, f in SCORING_EQUATIONS.items()},
          "alpha =", softmax([0, 0, 0]).round(4).tolist())
    for u in UNITS:
        c = CREATURES[u]
        print(f"  {u:10s} ({c.virtue:10s}) need={r.needs[u]:.3f} h={r.activations[u]:.3f} "
              f"z*={r.harmonized[u]:.3f} consensus={r.consensus[u]:.3f} alloc={r.allocation[u]:.3f}")
    print("  recommendation:", r.recommendation)
    print("  ", r.message)

    # DeGroot convergence on the creature graph, starting from the demo opinions
    O = opinions(np.array(list(r.harmonized.values())), 0.6)
    _, sp = degroot(HARMONY_W, O, steps=60)
    _, sp_lazy = degroot(HARMONY_W_LAZY, O, steps=60)
    steps_to_1e6 = next(k for k, s in enumerate(sp) if s < 1e-6)
    eig = np.sort(np.abs(np.linalg.eigvals(HARMONY_W)))[::-1]
    W4 = np.linalg.matrix_power(HARMONY_W, 4)
    with open(os.path.join(OUT, "degroot.dat"), "w") as f:
        f.write("t spread spreadlazy fj\n")
        for k in range(len(sp)):
            f.write(f"{k} {max(sp[k],1e-17):.6e} {max(sp_lazy[k],1e-17):.6e} {max(r.fj_residuals[k],1e-17):.6e}\n")

    # crisis gate check
    crisis = run_pipeline(x, text="I want to die")

    # Seeded random-graph experiment: empirical contraction vs Lemma bound
    rng = np.random.default_rng(2026)
    rows = []
    for n in (4, 8, 16):
        ratios, bounds, steps = [], [], []
        for _ in range(200):
            A = np.zeros((n, n)); perm = rng.permutation(n)
            for k in range(n):
                A[perm[k], perm[(k + 1) % n]] = 1
            A = np.maximum(A, (rng.random((n, n)) < 0.3)); np.fill_diagonal(A, 1)
            W = A * rng.uniform(0.05, 1, (n, n)); W /= W.sum(1, keepdims=True)
            p = rng.random(n)
            Wk = np.linalg.matrix_power(W, n - 1)
            ratios.append(np.ptp(Wk @ p) / np.ptp(p))
            bounds.append(1 - n * Wk.min())
            _, s = degroot(W, p, steps=20000, tol=1e-9 * np.ptp(p))
            steps.append(len(s) - 1)
        rows.append({"n": n, "mean_ratio": float(np.mean(ratios)), "max_ratio": float(np.max(ratios)),
                     "mean_bound": float(np.mean(bounds)),
                     "violations": int(np.sum(np.array(ratios) > np.array(bounds) + 1e-12)),
                     "median_steps": float(np.median(steps)), "max_steps": int(np.max(steps))})

    res = {
        "example_input": EXAMPLE, "psi": r.psi, "daily": r.daily, "blended_daily": r.blended_daily,
        "equations": {k: f(x) for k, f in SCORING_EQUATIONS.items()},
        "needs": r.needs, "activations": r.activations, "harmonized": r.harmonized,
        "consensus": r.consensus, "allocation": r.allocation, "recommendation": r.recommendation,
        "degroot_spreads_first": sp[:6], "degroot_spread_30": sp[30], "degroot_spread_60": sp[60],
        "degroot_lazy_spread_30": sp_lazy[30],
        "steps_to_1e-6": steps_to_1e6, "second_eig_modulus": float(eig[1]),
        "lazy_second_eig_modulus": float(np.sort(np.abs(np.linalg.eigvals(HARMONY_W_LAZY)))[::-1][1]),
        "W4_min": float(W4.min()), "dobrushin_W": dobrushin(HARMONY_W), "dobrushin_W4": dobrushin(W4),
        "pi_metropolis": consensus_weights(HARMONY_W).tolist(),
        "pi_lazy": consensus_weights(HARMONY_W_LAZY).tolist(),
        "fj_residual_0": r.fj_residuals[0], "fj_residual_10": r.fj_residuals[10],
        "fj_residual_20": r.fj_residuals[20], "fj_residual_40": r.fj_residuals[40],
        "crisis_gate": {"crisis": crisis.crisis, "daily": crisis.daily},
        "constant_residuals": check_constant_identities(),
        "random_graphs": rows,
        "W": HARMONY_W.tolist(),
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"  DeGroot spread: t=0 {sp[0]:.4f}, t=10 {sp[10]:.3e}, t=30 {sp[30]:.3e}; "
          f"steps to <1e-6: {steps_to_1e6}; |lambda_2| = {eig[1]:.4f}")
    print(f"  Banach residual: k=0 {r.fj_residuals[0]:.3e}, k=10 {r.fj_residuals[10]:.3e}, k=20 {r.fj_residuals[20]:.3e}")
    for row in rows:
        print("  random graphs:", row)
    print(f"  crisis gate: crisis={crisis.crisis}, score suppressed={crisis.daily is None}")


if __name__ == "__main__":
    main()
