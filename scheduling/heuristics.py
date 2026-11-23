"""
Priority rules for Pm | r_i = 0 | sum_i w_i U_i.

A rule is a pure function  fn(f, c) -> np.ndarray  of length n, where `f` is the
per-job feature table and `c` the instance-level scalars (see environment.py).
A HIGHER score means "more worth keeping on time"; the decoder does the rest.

Two groups:
  week1  -- the nine classical rules benchmarked in the Week-1 report.
  week2  -- rules added afterwards to probe specific questions (does the
            direction of a signal matter? does congestion information help?
            does the evict-decoder rescue a weak signal? what does zero
            information score?).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

EPS = 1e-9


@dataclass(frozen=True)
class Rule:
    fn: Callable                # (features, context) -> priority array
    decoder: str                # "greedy_insert" | "mh_drop"
    group: str                  # "week1" | "week2"
    formula: str                # human-readable priority expression
    idea: str                   # one-line rationale


# --------------------------------------------------------------------------- #
# Week 1: the nine classical rules
# --------------------------------------------------------------------------- #
RULES: Dict[str, Rule] = {
    "WSPT": Rule(
        fn=lambda f, c: f["wp"],
        decoder="greedy_insert", group="week1",
        formula="w_i / p_i",
        idea="Value per unit of machine time: the classic weighted-shortest-"
             "processing-time ratio rule.",
    ),
    "WSPT-EDD": Rule(
        fn=lambda f, c: f["wp"] - 0.001 * f["d_norm"],
        decoder="greedy_insert", group="week1",
        formula="w_i / p_i - 0.001 * (d_i / max d)",
        idea="WSPT with a tiny earliest-deadline tie-break, so equal-density "
             "jobs are taken in deadline order.",
    ),
    "MooreHodgson-W": Rule(
        fn=lambda f, c: f["wp"],
        decoder="mh_drop", group="week1",
        formula="w_i / p_i   (evict decoder)",
        idea="Same density signal as WSPT but decoded by admit-then-evict; "
             "recovers the classical weighted Moore-Hodgson algorithm.",
    ),
    "Density-wpd": Rule(
        fn=lambda f, c: f["dens"],
        decoder="greedy_insert", group="week1",
        formula="w_i / (p_i * d_i)",
        idea="Charges a job for its deadline as well as its length, so early-"
             "deadline jobs look cheaper.",
    ),
    "MaxWeight": Rule(
        fn=lambda f, c: f["w"],
        decoder="greedy_insert", group="week1",
        formula="w_i",
        idea="Pure greed: take the most valuable jobs first, ignore cost.",
    ),
    "MooreHodgson-U": Rule(
        fn=lambda f, c: -f["p"],
        decoder="mh_drop", group="week1",
        formula="-p_i   (evict decoder)",
        idea="Unweighted Moore-Hodgson: evict the longest job on overflow. "
             "Optimal when all weights are equal.",
    ),
    "SPT": Rule(
        fn=lambda f, c: -f["p"],
        decoder="greedy_insert", group="week1",
        formula="-p_i",
        idea="Shortest job first: maximises the *count* of on-time jobs, "
             "blind to value.",
    ),
    "EDD": Rule(
        fn=lambda f, c: -f["d"],
        decoder="greedy_insert", group="week1",
        formula="-d_i",
        idea="Earliest deadline first: the feasibility ordering, used here as "
             "an admission priority.",
    ),
    "Slack": Rule(
        fn=lambda f, c: f["slack"],
        decoder="greedy_insert", group="week1",
        formula="d_i - p_i",
        idea="Largest slack first: prefers jobs with the most spare room. "
             "Included as the week-1 slack variant.",
    ),

    # ----------------------------------------------------------------------- #
    # Week 2: added to probe specific questions
    # ----------------------------------------------------------------------- #
    "MinSlack": Rule(
        fn=lambda f, c: -f["slack"],
        decoder="greedy_insert", group="week2",
        formula="-(d_i - p_i)",
        idea="Least slack first, the standard dispatching form. Directly tests "
             "whether week-1's Slack rule was simply pointed the wrong way.",
    ),
    "LPT": Rule(
        fn=lambda f, c: f["p"],
        decoder="greedy_insert", group="week2",
        formula="p_i",
        idea="Longest job first: the deliberate inverse of SPT, included as a "
             "direction control.",
    ),
    "WSPT-Congestion": Rule(
        fn=lambda f, c: f["wp"] / (1.0 + f["cum_p_before"]
                                   / np.maximum(c["m"] * f["d"], EPS)),
        decoder="greedy_insert", group="week2",
        formula="(w_i / p_i) / (1 + C_i),  C_i = (work due by d_i) / (m * d_i)",
        idea="WSPT discounted by how saturated the job's own deadline prefix "
             "already is under Horn's condition.",
    ),
    "WSPT-Laxity": Rule(
        fn=lambda f, c: f["wp"] * (1.0 + f["laxity"]),
        decoder="greedy_insert", group="week2",
        formula="(w_i / p_i) * (1 + (d_i - p_i) / d_i)",
        idea="WSPT boosted by relative laxity: among equally dense jobs, prefer "
             "the one that is easier to place.",
    ),
    "Value-per-Slack": Rule(
        fn=lambda f, c: f["w"] / (1.0 + np.maximum(f["slack"], 0.0)),
        decoder="greedy_insert", group="week2",
        formula="w_i / (1 + max(d_i - p_i, 0))",
        idea="Value per unit of spare room: a scarcity-priced alternative to "
             "value per unit of machine time.",
    ),
    "MH-Density": Rule(
        fn=lambda f, c: f["dens"],
        decoder="mh_drop", group="week2",
        formula="w_i / (p_i * d_i)   (evict decoder)",
        idea="Second same-signal / different-decoder probe, repeating the "
             "WSPT vs Moore-Hodgson-W comparison on the density signal.",
    ),
    "MH-MinSlack": Rule(
        fn=lambda f, c: -f["slack"],
        decoder="mh_drop", group="week2",
        formula="-(d_i - p_i)   (evict decoder)",
        idea="Tests whether the evict decoder can rescue a weight-blind signal "
             "that fails badly under greedy insertion.",
    ),
    "Random": Rule(
        fn=lambda f, c: np.random.default_rng(12345).random(f["p"].shape[0]),
        decoder="greedy_insert", group="week2",
        formula="uniform random in [0,1)",
        idea="Zero-information control: fixes the floor that any real signal "
             "must beat. Seeded, so it is reproducible.",
    ),
}

WEEK1 = [k for k, r in RULES.items() if r.group == "week1"]
WEEK2 = [k for k, r in RULES.items() if r.group == "week2"]
