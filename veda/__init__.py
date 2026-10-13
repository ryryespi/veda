"""Project Veda: a unified mathematical framework for a personal reflection OS.

All credit and glory to God. I owe my knowledge to my Lord Jesus Christ.

The four subsystems are philosophical framings implemented as ordinary,
checkable mathematics. Nothing here measures or proves anything about
consciousness. Veda is a self-reflection aid, not therapy or diagnosis.
In a crisis in the US, call or text 988.
"""
from .veritas import CONSTANTS, VALUE_WEIGHTS, TERMS, Invariants
from .alpha import softmax, blend, daily_score, psi, daily_score_as_blend, blended_daily
from .mental import CREATURES, CREATURE_DAG, needs, message_pass, topological_order
from .peace import (influence_matrix, metropolis_matrix, dobrushin, strongly_connected, is_row_stochastic, degroot, consensus_weights, nash_bargain,
                    friedkin_johnsen_fixed_point, fj_iterate)
from .pipeline import run_pipeline, PipelineResult
from .safety import crisis_check, CRISIS_MESSAGE

__all__ = [n for n in dir() if not n.startswith("_")]
__version__ = "0.1.0"
