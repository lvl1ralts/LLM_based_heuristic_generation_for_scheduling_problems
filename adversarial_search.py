#!/usr/bin/env python3
"""
Adversarial instance search for WSPT on 1 | r_i = 0 | sum w_i U_i.

Follows the MetaOpt framing (Namyar et al.): the outer problem chooses the
instance, the inner problems run the heuristic and the exact optimum, and we
maximise the gap between them. We use the single-machine case throughout
because the Lawler-Moore DP gives a true optimum there, so the objective being
maximised is an exact gap rather than a gap to a relaxation.

Three experiments (superseded by metaopt.py, which solves the search exactly):
  A  verify the hand-derived two-job construction
  C  constrained-box search: same value ranges as the generator, but arbitrary
     structure. Hill climbing on the gap.
  D  unconstrained: let the weight range grow, and watch the ratio diverge.

Writes results/adversarial_*.csv and prints a summary.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduling.environment import (accepted_mask, evaluate,
                                    exact_single_machine, lower_bound)
from scheduling.heuristics import RULES
from scheduling.instances import P1Instance, gen_p1

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
EPS = 1e-9
WSPT = RULES["WSPT"].fn


# --------------------------------------------------------------------------- #
def make(p, d, w, m=1, name="adv"):
    return P1Instance(p=np.asarray(p, float), d=np.asarray(d, float),
                      w=np.asarray(w, float), m=m, name=name)


def score(inst):
    """Return (heuristic objective, true optimum, normalised gap %, ratio)."""
    heur = evaluate(inst, WSPT, "greedy_insert")
    opt = exact_single_machine(inst)
    W = inst.total_weight
    gap = 100.0 * (heur - opt) / max(W - opt, EPS)
    ratio = (heur / opt) if opt > EPS else float("inf") if heur > EPS else 1.0
    return heur, opt, gap, ratio


def integerise(inst):
    inst.d = np.rint(inst.d)
    inst.meta.pop("_p1feat", None)
    inst.meta.pop("_lb", None)
    return inst


# --------------------------------------------------------------------------- #
def experiment_A():
    print("=" * 74)
    print("A. the hand-derived two-job construction")
    print("=" * 74)
    rows = []
    for C in (10, 100, 1000, 10000):
        inst = make(p=[1, C], d=[C, C], w=[2, C])
        heur, opt, gap, ratio = score(inst)
        acc = accepted_mask(inst, WSPT, "greedy_insert")
        rows.append(dict(C=C, wspt_obj=heur, opt_obj=opt, gap_pct=gap, ratio=ratio))
        print(f"  C={C:>6}  jobs A(p=1,w=2,d={C}) B(p={C},w={C},d={C})")
        print(f"          WSPT keeps {list(np.where(acc)[0])}  ->  obj={heur:>8.0f}"
              f"   OPT obj={opt:>6.0f}   gap={gap:6.2f}%   ratio={ratio:7.1f}x")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "adversarial_construction.csv"),
                              index=False)
    print()


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
def climb(n, p_hi, w_hi, rng, iters=4000):
    """Hill-climb integer (p, d, w) to maximise the normalised gap."""
    p = rng.integers(1, p_hi + 1, n).astype(float)
    w = rng.integers(1, w_hi + 1, n).astype(float)
    d_hi = max(int(p.sum()), int(p.max()) + 1)
    d = rng.integers(1, d_hi + 1, n).astype(float)
    d = np.maximum(d, p)                       # keep every job individually feasible
    cur = score(make(p, d, w))[2]
    for _ in range(iters):
        i = int(rng.integers(n))
        which = int(rng.integers(3))
        p2, d2, w2 = p.copy(), d.copy(), w.copy()
        step = float(rng.choice([-20, -5, -1, 1, 5, 20]))
        if which == 0:
            p2[i] = min(max(p2[i] + step, 1), p_hi)
            d2[i] = max(d2[i], p2[i])
        elif which == 1:
            d2[i] = min(max(d2[i] + step, p2[i]), max(int(p2.sum()), 1))
        else:
            w2[i] = min(max(w2[i] + step, 1), w_hi)
        cand = score(make(p2, d2, w2))[2]
        if cand >= cur:
            p, d, w, cur = p2, d2, w2, cand
    return p, d, w, cur


def experiment_C(n=10, restarts=60, p_hi=100, w_hi=10):
    print("=" * 74)
    print(f"C. constrained-box search  (p in [1,{p_hi}], w in [1,{w_hi}], "
          f"n={n}, m=1)")
    print("=" * 74)
    rng = np.random.default_rng(7)
    best = None
    for r in range(restarts):
        p, d, w, gap = climb(n, p_hi, w_hi, rng)
        if best is None or gap > best[3]:
            best = (p, d, w, gap)
    p, d, w, gap = best
    inst = make(p, d, w)
    heur, opt, gap, ratio = score(inst)
    acc = accepted_mask(inst, WSPT, "greedy_insert")
    order = np.argsort(-(w / p), kind="stable")
    print(f"  best found: gap={gap:.2f}%   ratio={ratio:.2f}x   "
          f"WSPT={heur:.0f}  OPT={opt:.0f}  W={inst.total_weight:.0f}")
    print(f"  {'job':>4} {'p':>5} {'d':>6} {'w':>4} {'w/p':>6}  WSPT")
    for i in order:
        print(f"  {i:>4} {p[i]:>5.0f} {d[i]:>6.0f} {w[i]:>4.0f} "
              f"{w[i]/p[i]:>6.2f}  {'keep' if acc[i] else 'DROP'}")
    pd.DataFrame(dict(p=p, d=d, w=w, wspt_keep=acc)).to_csv(
        os.path.join(RESULTS, "adversarial_box_best.csv"), index=False)
    print()
    return gap


# --------------------------------------------------------------------------- #
def experiment_D(n=10, restarts=25):
    print("=" * 74)
    print("D. relaxing the weight range")
    print("=" * 74)
    rng = np.random.default_rng(11)
    rows = []
    for w_hi in (10, 50, 200, 1000):
        best = None
        for r in range(restarts):
            p, d, w, gap = climb(n, 100, w_hi, rng, iters=3000)
            if best is None or gap > best[3]:
                best = (p, d, w, gap)
        p, d, w, _ = best
        heur, opt, gap, ratio = score(make(p, d, w))
        rows.append(dict(w_max=w_hi, gap_pct=gap, ratio=ratio,
                         wspt=heur, opt=opt))
        print(f"  w in [1,{w_hi:>4}]  ->  gap {gap:6.2f}%   ratio {ratio:6.2f}x"
              f"   (WSPT {heur:.0f} vs OPT {opt:.0f})")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "adversarial_weightrange.csv"),
                              index=False)
    print()


if __name__ == "__main__":
    experiment_A()
    gc = experiment_C()
    experiment_D()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  benchmark reported WSPT at 0.12% from optimum (m=1 family, 8 instances)")
    print(f"  constrained-box worst (same ranges): {gc:.2f}%")
    print(f"  -> results in {RESULTS}")
