# Project Veda

*All credit and glory to God. I owe my knowledge to my Lord Jesus Christ.*

Project Veda is a unified mathematical framework for a personal reflection
operating system (Ryan Espinosa, October 2026). It brings four systems together
as a pipeline a computer can run and a reader can check. The paper is
[Project_Veda_Paper.pdf](Project_Veda_Paper.pdf)
(source: [Project_Veda_Paper.tex](Project_Veda_Paper.tex)).

The four systems are **philosophical framings**. The names and pictures do not
model, measure, or prove anything about consciousness, the soul, or
metaphysical truth. What the package does is turn each system into ordinary
mathematics and **prove real properties** of that mathematics. The Python
package `veda` implements every construction, and the test suite checks each
theorem numerically.

| Module | System | What is proved |
|---|---|---|
| `veda/veritas.py` | **Veritas Aeterna** | Fixed constants, a value-weight vector on the simplex, and updates that stay inside the invariant set (Theorem 1) |
| `veda/alpha.py` | **Project Alpha** | Softmax blends of component equations; MindFrame Ψ and the Daily Score as a convex blend (Theorems 2–4, Proposition 1) |
| `veda/mental.py` | **Mental Project** | Message passing on a creature DAG: one Kahn pass terminates, stays in [0, 1], and does not amplify differences (Theorems 5–6) |
| `veda/peace.py` | **Peaceful Division** | DeGroot consensus, Friedkin–Johnsen contraction, and weighted Nash bargaining (Lemmas 1–2, Theorems 7–9, Corollary 1) |
| `veda/pipeline.py` | Integration | The full pipeline is bounded and its iteration converges (Theorem 10) |
| `veda/safety.py` | Crisis gate | Scoring stops and only support resources are returned (US: 988) |

## The four systems

**Veritas Aeterna** (“eternal truth”) holds things that should not drift:
real mathematical constants (π, e, φ, √2) with checkable identities, and a
normalized value-weight vector on the probability simplex. Admissible updates
(box projection, simplex projection, convex mixing, exponentiated gradient)
preserve the invariant set.

**Project Alpha** blends component equations with softmax weights. The blend
stays between its components and is Lipschitz in both the inputs and the
weights. The MindFrame Ψ index,

Ψ = 0.18 C + 0.14 S + 0.12 M − 0.26 R − 0.31 B,

is an exact instance: its Daily Score is a convex blend of oriented terms
(calm, support, skills use, avoidance, thought believability).

**Mental Project** pictures mental architecture as five invented creature
units on a directed acyclic graph. Each creature is a mnemonic for a
processing unit that reads one MindFrame term and suggests one skill:

| Creature | Picture | Virtue | Term |
|---|---|---|---|
| Lumowl | a lantern-eyed night watcher | reflection | thought believability |
| Brakkor | a stone-maned beast that walks toward the storm | courage | avoidance |
| Mistfawn | a fog spirit that slows every breath | calm | calm |
| Hivewright | a many-armed builder laying one brick a day | discipline | skills use |
| Tidelyn | a sea serpent linking far shores | connection | support |

A single Kahn pass evaluates each unit once. Activations stay in [0, 1], and
the map from needs to activations is non-expansive.

**Peaceful Division** settles disagreement among those units. DeGroot
averaging converges to consensus. Anchored Friedkin–Johnsen averaging is a
contraction with a unique fixed point (Banach). A weighted Nash bargain
divides an attention budget with a unique, Pareto-optimal closed form.

## Install

```bash
pip install -e .
```

That installs the `veda` package and its runtime dependency (NumPy). Tests
also need pytest and Hypothesis, declared as the `test` extra:

```bash
pip install -e ".[test]"
```

Python 3.10 or newer is required.

## Tests

```bash
pytest
```

The suite checks the theorems above, including randomized property tests.
From a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/pytest -v
```

## Demo

```bash
python demo.py
```

The demo runs the pipeline on made-up illustration values (not data from any
person), prints Ψ, the Daily Score, each creature’s need and allocation, and
writes `paper/results.json` and `paper/degroot.dat`. Regenerating the paper
tables from those outputs:

```bash
python paper/make_results.py
pdflatex Project_Veda_Paper.tex && pdflatex Project_Veda_Paper.tex
```

## Safety

Veda is a self-reflection aid, not therapy or diagnosis. If the text contains
crisis language, scoring is suspended and the only output is support
resources. In a crisis in the US, call or text 988 (or 911 in an emergency).
Outside the US, contact your local emergency number.
