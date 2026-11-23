#!/usr/bin/env python3
"""
The eight similarity cases: each of processing time (p), deadline (d) and
weight (w) is either "similar" (within +-10% of a centre, never all equal) or
"varied" (the usual wide spread). 2 x 2 x 2 = 8 cases.

  python similarity_cases.py

Every rule in RULES is run on every case and compared with the exact optimum:
the Lawler-Moore DP on one machine, the MILP from metaopt.exact_opt (HiGHS) on
several machines.

Writes to results/similarity/:
  gaps_by_case.csv   rule x case, mean % gap to the optimum
  detail.csv         one row per (setting, case, rule): mean, worst, optimal-on
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metaopt import exact_opt
from scheduling.environment import evaluate_suite, exact_single_machine
from scheduling.heuristics import RULES
from scheduling.instances import P1Instance

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "similarity")
os.makedirs(OUT, exist_ok=True)
EPS = 1e-9

SPREAD = 0.10      # "similar" = centre * U[0.9, 1.1]
T = 0.5            # tardiness factor: deadlines centred on half the per-machine load
R_VARIED = 0.6     # due-date range when deadlines are varied (generator default)
SETTINGS = [       # (label, n, m, instances per case)
    ("m=1, n=50", 50, 1, 30),
    ("m=4, n=40", 40, 4, 20),
]


def gen_case(n, m, seed, p_sim, d_sim, w_sim):
    rng = np.random.default_rng(seed)
    if p_sim:
        p = np.rint(50 * rng.uniform(1 - SPREAD, 1 + SPREAD, n))
    else:
        p = rng.integers(1, 101, n).astype(float)
    if w_sim:
        w = np.rint(50 * rng.uniform(1 - SPREAD, 1 + SPREAD, n))
    else:
        w = rng.integers(1, 101, n).astype(float)
    centre = p.sum() / m * (1 - T)
    half = SPREAD if d_sim else R_VARIED / 2 / (1 - T)   # half-width relative to centre
    lo = max(centre * (1 - half), p.max())
    hi = max(centre * (1 + half), lo + 1)
    d = np.rint(rng.uniform(lo, hi, n))
    d = np.maximum(d, p)
    return P1Instance(p=p, d=d, w=w, m=m, name=f"sim_s{seed}")


def case_name(p_sim, d_sim, w_sim):
    tag = [x for x, s in (("p", p_sim), ("d", d_sim), ("w", w_sim)) if s]
    return "none similar" if not tag else " & ".join(tag) + " similar"


def main():
    t0 = time.time()
    cases = list(itertools.product([False, True], repeat=3))
    cases.sort(key=lambda c: (sum(c), [not x for x in c]))
    detail = []
    for s_i, (label, n, m, k) in enumerate(SETTINGS):
        for c_i, (ps, ds, ws) in enumerate(cases):
            cname = case_name(ps, ds, ws)
            insts = [gen_case(n, m, 30000 + 1000 * s_i + 100 * c_i + j, ps, ds, ws)
                     for j in range(k)]
            opt, proved = [], 0
            for inst in insts:
                if m == 1:
                    opt.append(exact_single_machine(inst)); proved += 1
                else:
                    v, _, ok = exact_opt(inst, time_limit=60)
                    opt.append(v); proved += ok
            opt = np.array(opt)
            W = np.array([i.total_weight for i in insts])
            band = np.maximum(W - opt, EPS)
            for name, rule in RULES.items():
                objs = evaluate_suite(insts, rule.fn, rule.decoder)
                g = 100 * (objs - opt) / band
                detail.append(dict(setting=label, case=cname, rule=name,
                                   mean_gap=g.mean(), worst_gap=g.max(),
                                   optimal_on=int((objs <= opt + 1e-6).sum()),
                                   n_inst=k, opt_proved=proved,
                                   opt_loss_pct=100 * (opt / W).mean()))
            print(f"  {label:10s} {cname:24s} opt proved {proved}/{k}  "
                  f"({time.time()-t0:6.1f}s)", flush=True)

    df = pd.DataFrame(detail)
    df.round(3).to_csv(os.path.join(OUT, "detail.csv"), index=False)
    wide = df.pivot_table(index="rule", columns=["setting", "case"],
                          values="mean_gap", sort=False)
    wide.round(2).to_csv(os.path.join(OUT, "gaps_by_case.csv"))

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for label, *_ in SETTINGS:
        sub = df[df.setting == label].pivot_table(index="rule", columns="case",
                                                  values="mean_gap", sort=False)
        sub = sub[[case_name(*c) for c in cases]]
        sub["avg"] = sub.mean(axis=1)
        print(f"\n=== {label}: mean % gap to the exact optimum ===")
        print(sub.sort_values("avg").round(2).to_string())
    print(f"\ndone in {time.time()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
