"""
Instance representation and generator for Pm | r_i = 0 | sum_i w_i U_i.

Brucker, "Scheduling Algorithms": Ch.1 (alpha|beta|gamma classification) fixes the
notation; Ch.5 (parallel machines) gives the structural results we lean on, in
particular Horn's feasibility condition for P|pmtn, r_i=0, d_i| -- used here as a
cheap and valid relaxation-based lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


@dataclass
class P1Instance:
    """n independent jobs, m identical parallel machines, all released at t=0."""

    p: np.ndarray  # (n,) processing times            > 0
    d: np.ndarray  # (n,) deadlines (due dates)       > 0
    w: np.ndarray  # (n,) weights                     > 0
    m: int
    name: str = "p1"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.p.shape[0])

    @property
    def total_weight(self) -> float:
        return float(self.w.sum())

    def __post_init__(self):
        self.p = np.asarray(self.p, dtype=np.float64)
        self.d = np.asarray(self.d, dtype=np.float64)
        self.w = np.asarray(self.w, dtype=np.float64)
        assert self.p.shape == self.d.shape == self.w.shape
        assert self.m >= 1


def gen_p1(
    n: int = 50,
    m: int = 4,
    seed: int = 0,
    tardiness_factor: float = 0.4,
    due_range: float = 0.6,
    weight_model: str = "uncorrelated",
    p_low: int = 1,
    p_high: int = 100,
    p_model: str = "uniform",
) -> P1Instance:
    """Potts & Van Wassenhove style generator, adapted to m identical machines.

    tardiness_factor T controls how many jobs are tight; due_range R the spread.
    d_i ~ U[ Pbar (1 - T - R/2),  Pbar (1 - T + R/2) ]  with Pbar = sum(p)/m.
    """
    rng = np.random.default_rng(seed)
    if p_model == "uniform":
        p = rng.integers(p_low, p_high + 1, size=n).astype(float)
    elif p_model == "bimodal":  # many tiny jobs plus a few very long ones
        small = rng.integers(p_low, max(p_low + 1, p_high // 8), size=n).astype(float)
        large = rng.integers(int(0.7 * p_high), p_high + 1, size=n).astype(float)
        p = np.where(rng.random(n) < 0.75, small, large)
    elif p_model == "lognormal":
        p = np.clip(np.round(rng.lognormal(3.0, 0.9, size=n)), p_low, 4 * p_high).astype(float)
    else:
        raise ValueError(p_model)
    pbar = p.sum() / m
    lo = pbar * (1.0 - tardiness_factor - due_range / 2.0)
    hi = pbar * (1.0 - tardiness_factor + due_range / 2.0)
    lo = max(lo, float(p.max()))  # keep every job individually schedulable
    hi = max(hi, lo + 1.0)
    d = rng.uniform(lo, hi, size=n)
    d = np.maximum(d, p)  # p_i <= d_i, otherwise job i can never be on time

    if weight_model == "uncorrelated":
        w = rng.integers(1, 11, size=n).astype(float)
    elif weight_model == "correlated":  # weight grows with processing time (harder)
        w = np.round(p / 10.0 + rng.integers(0, 4, size=n)).astype(float) + 1.0
    elif weight_model == "anticorrelated":  # short jobs are valuable
        w = np.round((p_high - p) / 10.0 + rng.integers(0, 4, size=n)).astype(float) + 1.0
    else:
        raise ValueError(weight_model)

    return P1Instance(
        p=p,
        d=d,
        w=w,
        m=m,
        name=f"p1_n{n}_m{m}_T{tardiness_factor}_R{due_range}_{weight_model}_{p_model}_s{seed}",
        meta=dict(n=n, m=m, seed=seed, T=tardiness_factor, R=due_range,
                  weights=weight_model, p_model=p_model),
    )


def p1_suite(spec: Sequence[dict]) -> List[P1Instance]:
    return [gen_p1(**s) for s in spec]
