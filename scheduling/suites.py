"""
Held-out benchmark instance families.

WEEK1_SUITES reproduces the five families used in the Week-1 report, with the
same seeds, so those numbers stay comparable. WEEK2_SUITES adds seven families
that probe conditions the Week-1 suite never touched: loose deadlines, the two
non-uniform weight models, a narrow due-date window, high machine counts, the
single-machine case (where an exact optimum is computable), and a larger scale.

All seed blocks are disjoint from each other and from the 0-17 / 500-517 seeds
used for tuning, so nothing here has ever been fitted on.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from scheduling.instances import P1Instance, gen_p1

# the one family where m = 1, so the Lawler-Moore DP gives a true optimum
EXACT_SUITE = "single machine m=1"


def _integerise(inst: P1Instance) -> P1Instance:
    """Round deadlines to integers so the exact DP and the heuristics see the
    identical instance (the DP rounds internally; p is already integral)."""
    inst.d = np.rint(inst.d)
    inst.meta.pop("_p1feat", None)
    inst.meta.pop("_lb", None)
    return inst


def week1_suites() -> Dict[str, List[P1Instance]]:
    S = {}
    S["in-dist n50-100"] = [gen_p1(n=n, m=m, seed=1000 + s, tardiness_factor=T)
                            for n in (50, 100) for m in (4, 8)
                            for T in (0.4,) for s in range(3)]
    S["scale-shift n200-300"] = [gen_p1(n=n, m=m, seed=2000 + s, tardiness_factor=0.5)
                                 for n in (200, 300) for m in (8, 12) for s in range(2)]
    S["shift: bimodal p"] = [gen_p1(n=120, m=5, seed=3000 + s, tardiness_factor=0.5,
                                    p_model="bimodal") for s in range(10)]
    S["shift: lognormal p"] = [gen_p1(n=120, m=5, seed=4000 + s, tardiness_factor=0.5,
                                      p_model="lognormal") for s in range(10)]
    S["shift: very tight T=0.8"] = [gen_p1(n=100, m=4, seed=5000 + s,
                                           tardiness_factor=0.8, due_range=0.3)
                                    for s in range(10)]
    return S


def week2_suites() -> Dict[str, List[P1Instance]]:
    S = {}
    S["loose T=0.1"] = [gen_p1(n=100, m=4, seed=6000 + s, tardiness_factor=0.1)
                        for s in range(8)]
    S["weights correlated"] = [gen_p1(n=100, m=4, seed=7000 + s, tardiness_factor=0.5,
                                      weight_model="correlated") for s in range(8)]
    S["weights anticorrelated"] = [gen_p1(n=100, m=4, seed=8000 + s, tardiness_factor=0.5,
                                          weight_model="anticorrelated") for s in range(8)]
    S["narrow due range R=0.1"] = [gen_p1(n=100, m=4, seed=9000 + s, tardiness_factor=0.5,
                                          due_range=0.1) for s in range(8)]
    S["many machines m=16"] = [gen_p1(n=200, m=16, seed=10000 + s, tardiness_factor=0.5)
                               for s in range(8)]
    S[EXACT_SUITE] = [_integerise(gen_p1(n=40, m=1, seed=11000 + s, tardiness_factor=0.5))
                      for s in range(8)]
    S["large scale n=500"] = [gen_p1(n=500, m=10, seed=12000 + s, tardiness_factor=0.5)
                              for s in range(4)]
    return S


def all_suites() -> Dict[str, List[P1Instance]]:
    S = week1_suites()
    S.update(week2_suites())
    return S


WEEK1_NAMES = list(week1_suites().keys())
WEEK2_NAMES = list(week2_suites().keys())
