"""Theorem 10 (integration) and safety gate."""
import numpy as np
from hypothesis import given

from veda.mental import UNITS, message_pass, needs
from veda.peace import friedkin_johnsen_fixed_point
from veda.pipeline import HARMONY_W, run_pipeline
from veda.safety import crisis_check
from strategies import terms5


@given(terms5)
def test_thm10_pipeline_bounded(x):
    r = run_pipeline(x)
    assert not r.crisis
    assert -1e-9 <= r.daily <= 100 + 1e-9 and -1e-9 <= r.blended_daily <= 100 + 1e-9
    a = np.array(list(r.allocation.values()))
    assert np.isclose(a.sum(), 1) and np.all(a >= 0.04 - 1e-12)
    b = np.array(list(r.consensus.values()))
    assert np.isclose(b.sum(), 1) and np.all(b > 0)
    assert all(0 <= v <= 1 + 1e-12 for v in r.harmonized.values())
    assert r.fj_residuals[-1] <= 0.5 ** 60 + 1e-12
    assert r.recommendation.split()[0] in UNITS


@given(terms5, terms5)
def test_thm10_settled_activation_is_1_lipschitz(x, x2):
    def zstar(v):
        tr = message_pass(needs(v))
        return friedkin_johnsen_fixed_point(HARMONY_W, [tr.h[u] for u in UNITS], 0.5)
    assert np.max(np.abs(zstar(x) - zstar(x2))) <= np.max(np.abs(x - x2)) + 1e-12


def test_crisis_gate_blocks_scoring():
    r = run_pipeline([0.5] * 5, text="sometimes I want to die")
    assert r.crisis and "988" in r.message and r.daily is None
    assert crisis_check("thinking about suicide")
    assert not crisis_check("I had a calm walk and called my sister")
    assert "988" in run_pipeline([0.5] * 5).message
