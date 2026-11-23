#!/usr/bin/env python3
"""
Run every priority rule against every held-out instance family.

  python run_benchmark.py

Writes to results/:
  benchmark_by_family.csv  rule x family, mean % gap to the LP lower bound
  benchmark_overall.csv    rule, overall mean % gap, decoder, group
  exact_gaps.csv           rule vs the true optimum on the single-machine family
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduling.environment import (evaluate_suite, exact_single_machine,
                                    lower_bound)
from scheduling.heuristics import RULES
from scheduling.suites import EXACT_SUITE, WEEK1_NAMES, all_suites

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)
EPS = 1e-9


def gap_pct(objs, lb, W):
    """Mean % of the 'losable' weight band that the rule actually loses."""
    return 100.0 * ((objs - lb) / np.maximum(W - lb, EPS)).mean()


def main():
    t0 = time.time()
    suites = all_suites()
    print(f"{len(suites)} families, "
          f"{sum(len(v) for v in suites.values())} instances, "
          f"{len(RULES)} rules\n")

    # ---- reference values per family -------------------------------------- #
    ref = {}
    for fam, insts in suites.items():
        lb = np.array([lower_bound(i) for i in insts])
        W = np.array([i.total_weight for i in insts])
        ref[fam] = (lb, W)
        print(f"  bounds done: {fam:24s} n_inst={len(insts):3d}  "
              f"mean W={W.mean():8.1f}  mean LB={lb.mean():7.2f}")

    # ---- every rule on every family --------------------------------------- #
    rows, raw = [], {}
    print()
    for name, rule in RULES.items():
        per_fam = {}
        for fam, insts in suites.items():
            objs = evaluate_suite(insts, rule.fn, rule.decoder)
            lb, W = ref[fam]
            per_fam[fam] = gap_pct(objs, lb, W)
            raw[(name, fam)] = objs
        overall = float(np.mean(list(per_fam.values())))
        rows.append(dict(rule=name, decoder=rule.decoder, group=rule.group,
                         overall=overall, **per_fam))
        print(f"  {name:18s} [{rule.group}] overall gap {overall:6.2f}%  "
              f"({time.time()-t0:5.1f}s)")

    df = pd.DataFrame(rows).sort_values("overall").reset_index(drop=True)

    # ---- portfolio oracle: best rule per instance -------------------------- #
    oracle = {}
    for fam, insts in suites.items():
        stack = np.stack([raw[(nm, fam)] for nm in RULES])
        best = stack.min(axis=0)
        lb, W = ref[fam]
        oracle[fam] = gap_pct(best, lb, W)
    oracle_row = dict(rule="[oracle: best rule per instance]", decoder="-",
                      group="analysis",
                      overall=float(np.mean(list(oracle.values()))), **oracle)

    fam_cols = list(suites.keys())
    out = pd.concat([df, pd.DataFrame([oracle_row])], ignore_index=True)
    out[["rule", "decoder", "group", "overall"] + fam_cols].round(3).to_csv(
        os.path.join(RESULTS, "benchmark_by_family.csv"), index=False)
    df[["rule", "decoder", "group", "overall"]].round(3).to_csv(
        os.path.join(RESULTS, "benchmark_overall.csv"), index=False)

    # ---- true optimality gap on the single-machine family ------------------ #
    insts = suites[EXACT_SUITE]
    opt = np.array([exact_single_machine(i) for i in insts])
    W = np.array([i.total_weight for i in insts])
    erows = []
    for name, rule in RULES.items():
        objs = raw[(name, EXACT_SUITE)]
        erows.append(dict(
            rule=name,
            gap_to_opt_pct=100.0 * ((objs - opt) / np.maximum(W - opt, EPS)).mean(),
            mean_excess_weight=float((objs - opt).mean()),
            optimal_on=int((objs <= opt + 1e-6).sum()),
            n_instances=len(insts),
        ))
    edf = pd.DataFrame(erows).sort_values("gap_to_opt_pct").reset_index(drop=True)
    edf.round(3).to_csv(os.path.join(RESULTS, "exact_gaps.csv"), index=False)

    # ---- where is there room left for a generated heuristic? --------------- #
    # best_single  : the best any one hand-written rule does on this family
    # oracle       : best rule chosen per instance (an upper bound on what rule
    #                *selection* alone could buy)
    # selection    : best_single - oracle  -> recoverable by picking per instance
    # structural   : oracle                -> still lost even with perfect choice
    opp = []
    for fam in suites:
        best_single = float(df[fam].min())
        best_rule = df.loc[df[fam].idxmin(), "rule"]
        orc = oracle[fam]
        opp.append(dict(family=fam, best_rule=best_rule, best_single_pct=best_single,
                        oracle_pct=orc, selection_gain_pct=best_single - orc,
                        structural_left_pct=orc))
    odf = pd.DataFrame(opp).sort_values("structural_left_pct", ascending=False)
    odf.round(3).to_csv(os.path.join(RESULTS, "llm_opportunity.csv"), index=False)

    # ---- console summary ---------------------------------------------------#
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== overall mean % gap to LP lower bound (lower is better) ===")
    print(out[["rule", "decoder", "group", "overall"]].round(2).to_string(index=False))

    print(f"\n=== week-1 families only (comparability check) ===")
    w1 = df.copy()
    w1["week1_only"] = w1[WEEK1_NAMES].mean(axis=1)
    print(w1.sort_values("week1_only")[["rule", "group", "week1_only"]]
          .round(2).to_string(index=False))

    print(f"\n=== vs TRUE optimum, {EXACT_SUITE} ({len(insts)} instances) ===")
    print(edf.round(2).to_string(index=False))

    print("\n=== room left for a generated heuristic, per family ===")
    print("  selection_gain = what per-instance rule switching would recover")
    print("  structural_left = still lost even with perfect rule choice\n")
    print(odf.round(2).to_string(index=False))

    print(f"\ndone in {time.time()-t0:.1f}s -> {RESULTS}")


if __name__ == "__main__":
    main()
