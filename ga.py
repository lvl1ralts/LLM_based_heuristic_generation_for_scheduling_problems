#!/usr/bin/env python3
"""
Genetic algorithm that evolves the priority AND the machine assignment together.

  python ga.py                         # all three experiments
  python ga.py adversarial similarity  # a subset (adversarial | similarity | benchmark)
  python ga.py --quick                 # small budget, for a smoke test

Every rule in this project so far is two steps: a priority score per job, then
a decoder that admits jobs in that order and puts each one on the earliest-free
machine (EDD list scheduling). The MetaOpt study found two kinds of failure:
ranking failures (the wrong jobs come first) and decoder failures (the optimal
job set does not pass the decoder's own feasibility test, so no priority can
reach it). A better priority can only fix the first kind.

Here one chromosome carries both decisions:

  keys[i]   in [0, 1]       priority of job i (random-key encoding)
  mach[i]   in {0..m-1}     machine job i should go on

Decoder (`decode_assign`): take jobs in decreasing key order; put job i on
mach[i] if that machine's job set stays on time (EDD on one machine is exact,
Jackson's rule), otherwise try the other machines in turn, otherwise reject.
Because a per-machine EDD test is exact, any optimal schedule can be written as
a chromosome (optimal jobs first, each on its optimal machine), so unlike the
list-scheduling decoder this one has no unreachable optimum.

Three variants are run with the same budget, to separate the two effects:

  GA-priority    evolves keys only, decoded by the existing greedy_insert
                 (EDD list scheduling), i.e. searches over priority rules' outputs
  GA-assignment  keys fixed to WSPT, evolves the machine assignment only
  GA-joint       evolves both together (the proposal)

The GA is a biased random-key GA (Goncalves & Resende 2011): elites survive,
offspring take each gene from the elite parent with probability rho, a share of
every generation is fresh random immigrants, plus point mutation. The initial
population is seeded with the schedules of the classical rules, so with elitism
the GA can never end worse than WSPT.

Writes to results/ga/:
  adversarial.csv    the 24 MetaOpt instances, every method vs the exact optimum
  similarity.csv     the eight similarity cases, m=1 (DP) and m=4 (MILP)
  benchmark.csv      the benchmark families, vs the LP bound (DP on m=1)
  summary.csv        mean gap per experiment x method
  convergence.csv    best-so-far per generation on a few instances
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduling.environment import (accepted_mask, evaluate, exact_single_machine,
                                    lower_bound, schedule_of)
from scheduling.heuristics import RULES
from scheduling.instances import P1Instance

try:
    from numba import njit
except ImportError:                      # pure Python still works, much slower
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "ga")
EPS = 1e-9

# rules whose schedules seed the initial population
SEED_RULES = ["WSPT", "WSPT-EDD", "WSPT-Laxity", "WSPT-Congestion", "MooreHodgson-W",
              "Density-wpd", "MaxWeight", "EDD", "SPT", "Value-per-Slack"]


# --------------------------------------------------------------------------- #
# decoders (numba-compiled when available)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def decode_assign(order, mach_pref, p, d, w, m):
    """Admit jobs in `order`; job i goes on mach_pref[i] if that machine stays
    on time under EDD, else on the next machine (cyclically) that does, else it
    is rejected. Returns (weight kept, machine per job, -1 if rejected)."""
    n = p.shape[0]
    dl = np.empty((m, n))                 # deadlines on each machine, EDD order
    cl = np.empty((m, n))                 # completion times, same order
    cnt = np.zeros(m, np.int64)
    mach = -np.ones(n, np.int64)
    kept = 0.0
    for t in range(n):
        i = order[t]
        if p[i] > d[i] + EPS:
            continue
        for r in range(m):
            j = (mach_pref[i] + r) % m
            c = cnt[j]
            k = 0
            while k < c and dl[j, k] <= d[i]:
                k += 1
            start = cl[j, k - 1] if k > 0 else 0.0
            if start + p[i] > d[i] + EPS:
                continue
            ok = True
            for s in range(k, c):         # every later job is pushed back by p_i
                if cl[j, s] + p[i] > dl[j, s] + EPS:
                    ok = False
                    break
            if not ok:
                continue
            for s in range(c, k, -1):
                dl[j, s] = dl[j, s - 1]
                cl[j, s] = cl[j, s - 1] + p[i]
            dl[j, k] = d[i]
            cl[j, k] = start + p[i]
            cnt[j] = c + 1
            mach[i] = j
            kept += w[i]
            break
    return kept, mach


@njit(cache=True)
def decode_list(order, p, d, w, m):
    """Same as environment.decode_greedy_insert: admit in `order` while the
    admitted set passes EDD list scheduling on the earliest-free machine."""
    n = p.shape[0]
    sd = np.empty(n + 1)
    sp = np.empty(n + 1)
    c = 0
    loads = np.empty(m)
    keep = np.zeros(n, np.bool_)
    kept = 0.0
    for t in range(n):
        i = order[t]
        if p[i] > d[i] + EPS:
            continue
        k = 0                             # bisect.insort_right on (d, p)
        while k < c and (sd[k] < d[i] or (sd[k] == d[i] and sp[k] <= p[i])):
            k += 1
        loads[:] = 0.0
        ok = True
        for s in range(c + 1):
            if s < k:
                ds, ps = sd[s], sp[s]
            elif s == k:
                ds, ps = d[i], p[i]
            else:
                ds, ps = sd[s - 1], sp[s - 1]
            j = np.argmin(loads)
            if loads[j] + ps > ds + EPS:
                ok = False
                break
            loads[j] += ps
        if ok:
            for s in range(c, k, -1):
                sd[s] = sd[s - 1]
                sp[s] = sp[s - 1]
            sd[k] = d[i]
            sp[k] = p[i]
            c += 1
            keep[i] = True
            kept += w[i]
    return kept, keep


# --------------------------------------------------------------------------- #
# GA
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GAConfig:
    mode: str = "joint"        # joint | priority | assignment
    pop: int = 100
    gens: int = 300
    elite: float = 0.15        # share of the population kept unchanged
    mutants: float = 0.10      # share replaced by random immigrants each generation
    rho: float = 0.70          # probability a gene comes from the elite parent
    pm_key: float = 0.05       # per-gene probability of perturbing a key
    pm_mach: float = 0.05      # per-gene probability of moving a job to a random machine
    sigma: float = 0.10        # key perturbation (keys are rank-scaled to [0, 1])
    patience: int = 150        # stop after this many generations without improvement
    seeded: bool = True        # start from the classical rules' schedules
    seed: int = 0


def _rank01(x):
    """Priority scores -> keys in [0, 1] by rank (stable), so mutation acts on
    position in the order, not on the scale of the rule's score."""
    n = len(x)
    order = np.argsort(-np.asarray(x, float), kind="stable")
    keys = np.empty(n)
    keys[order] = 1.0 - np.arange(n) / max(n - 1, 1)
    return keys


def _rule_machines(inst, rule, rng):
    """Machine of each job in the schedule a classical rule builds; random for
    the jobs it rejects."""
    keep = accepted_mask(inst, RULES[rule].fn, RULES[rule].decoder)
    mach = rng.integers(0, inst.m, inst.n)
    idx = np.where(keep)[0]
    rows = sorted((float(inst.d[i]), float(inst.p[i]), int(i)) for i in idx)
    for i, j, _, _ in schedule_of(rows, inst.m):
        mach[i] = j
    return mach


class GA:
    def __init__(self, inst: P1Instance, cfg: GAConfig):
        self.inst, self.cfg = inst, cfg
        self.p, self.d, self.w = inst.p, inst.d, inst.w
        self.m, self.n, self.W = inst.m, inst.n, inst.total_weight
        self.rng = np.random.default_rng(cfg.seed)
        self.wspt_keys = _rank01(RULES["WSPT-EDD"].fn(*_fc(inst)))
        self.evals = 0

    def fitness(self, keys, mach):
        """Weight lost (minimised)."""
        self.evals += 1
        order = np.argsort(-keys, kind="stable")
        if self.cfg.mode == "priority":
            kept, _ = decode_list(order, self.p, self.d, self.w, self.m)
        else:
            kept, _ = decode_assign(order, mach, self.p, self.d, self.w, self.m)
        return self.W - kept

    def _random(self, k):
        keys = self.rng.random((k, self.n))
        if self.cfg.mode == "assignment":
            keys[:] = self.wspt_keys
        return keys, self.rng.integers(0, self.m, (k, self.n))

    def _seeds(self):
        K, M = [], []
        for r in SEED_RULES:
            keys = _rank01(RULES[r].fn(*_fc(self.inst)))
            K.append(self.wspt_keys if self.cfg.mode == "assignment" else keys)
            M.append(_rule_machines(self.inst, r, self.rng))
        return np.array(K), np.array(M)

    def run(self):
        c, n = self.cfg, self.n
        n_el = max(1, int(round(c.elite * c.pop)))
        n_mu = int(round(c.mutants * c.pop))
        keys, mach = self._random(c.pop)
        if c.seeded:
            sk, sm = self._seeds()
            keys[:len(sk)], mach[:len(sm)] = sk, sm
        fit = np.array([self.fitness(keys[i], mach[i]) for i in range(c.pop)])
        best, history, stale = fit.min(), [], 0
        for g in range(c.gens):
            rank = np.argsort(fit, kind="stable")
            keys, mach, fit = keys[rank], mach[rank], fit[rank]
            history.append(float(fit[0]))
            if fit[0] < best - EPS:
                best, stale = fit[0], 0
            else:
                stale += 1
            if fit[0] <= EPS or stale >= c.patience:   # nothing late, or converged
                break
            n_off = c.pop - n_el - n_mu
            a = self.rng.integers(0, n_el, n_off)                  # elite parent
            b = self.rng.integers(n_el, c.pop, n_off)              # other parent
            from_a_k = self.rng.random((n_off, n)) < c.rho
            from_a_m = self.rng.random((n_off, n)) < c.rho         # independent coin
            ok = np.where(from_a_k, keys[a], keys[b])
            om = np.where(from_a_m, mach[a], mach[b])
            if c.mode != "assignment":
                hit = self.rng.random((n_off, n)) < c.pm_key
                ok = np.clip(ok + hit * self.rng.normal(0, c.sigma, (n_off, n)), 0, 1)
            if c.mode != "priority":
                hit = self.rng.random((n_off, n)) < c.pm_mach
                om = np.where(hit, self.rng.integers(0, self.m, (n_off, n)), om)
            rk, rm = self._random(n_mu)
            new_k = np.concatenate([keys[:n_el], ok, rk])
            new_m = np.concatenate([mach[:n_el], om, rm])
            new_f = np.concatenate([fit[:n_el],
                                    [self.fitness(new_k[i], new_m[i])
                                     for i in range(n_el, c.pop)]])
            keys, mach, fit = new_k, new_m, new_f
        i = int(np.argmin(fit))
        return dict(lost=float(fit[i]), keys=keys[i], mach=mach[i],
                    gens=len(history), evals=self.evals, history=history)


def _fc(inst):
    from scheduling.environment import context, features
    return features(inst), context(inst)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #
METHODS = ["GA-priority", "GA-assignment", "GA-joint"]
MODE = {"GA-priority": "priority", "GA-assignment": "assignment", "GA-joint": "joint"}


def baselines(inst):
    lost = {r: evaluate(inst, RULES[r].fn, RULES[r].decoder) for r in RULES}
    best_rule = min(lost, key=lost.get)
    return dict(WSPT=lost["WSPT"], best_rule=lost[best_rule], best_rule_name=best_rule)


def run_methods(inst, base_cfg, seeds, conv_rows=None, tag=None):
    """Every GA variant, `seeds` independent runs each. Returns mean and best lost."""
    out = {}
    for meth in METHODS:
        res = [GA(inst, replace(base_cfg, mode=MODE[meth], seed=s)).run() for s in seeds]
        L = [r["lost"] for r in res]
        out[meth] = float(np.mean(L))
        out[meth + " (best)"] = float(np.min(L))
        out[meth + " gens"] = float(np.mean([r["gens"] for r in res]))
        if conv_rows is not None:
            for g, v in enumerate(res[0]["history"]):
                conv_rows.append(dict(instance=tag, method=meth, gen=g, lost=v))
    return out


def gap(obj, ref, W):
    return 100.0 * (obj - ref) / max(W - ref, EPS)


def add_gaps(row, ref, W):
    for k in ["WSPT", "best_rule"] + METHODS + [m + " (best)" for m in METHODS]:
        row[k + " gap%"] = gap(row[k], ref, W)
    return row


def exp_adversarial(cfg, seeds, conv):
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "results", "metaopt", "*.json"))):
        J = json.load(open(path))
        inst = P1Instance(p=J["p"], d=J["d"], w=J["w"], m=J["m"],
                          name=os.path.basename(path)[:-5])
        row = dict(instance=inst.name, family=J["family"], vs=J["encoded_rule"],
                   n=inst.n, m=inst.m, cause=J["cause"], W=inst.total_weight,
                   opt=J["opt_lost"])
        row.update(baselines(inst))
        row.update(run_methods(inst, cfg, seeds, conv, inst.name))
        rows.append(add_gaps(row, row["opt"], row["W"]))
        print(f"  adv {inst.name:32s} opt {row['opt']:5.1f}  WSPT {row['WSPT']:5.1f}  "
              + "  ".join(f"{m} {row[m]:5.1f}" for m in METHODS), flush=True)
    return pd.DataFrame(rows)


def exp_similarity(cfg, seeds, conv, per_case, milp_limit):
    import itertools
    from metaopt import exact_opt
    from similarity_cases import SETTINGS, case_name, gen_case
    cases = list(itertools.product([False, True], repeat=3))
    cases.sort(key=lambda c: (sum(c), [not x for x in c]))
    rows = []
    for s_i, (label, n, m, _) in enumerate(SETTINGS):
        for c_i, (ps, ds, ws) in enumerate(cases):
            for j in range(per_case):             # same seeds as similarity_cases.py
                inst = gen_case(n, m, 30000 + 1000 * s_i + 100 * c_i + j, ps, ds, ws)
                if m == 1:
                    opt, proved = exact_single_machine(inst), True
                else:
                    opt, _, proved = exact_opt(inst, time_limit=milp_limit)
                row = dict(setting=label, case=case_name(ps, ds, ws), instance=inst.name,
                           n=n, m=m, W=inst.total_weight, opt=opt, opt_proved=bool(proved))
                row.update(baselines(inst))
                tag = f"{label}|{row['case']}|{j}"
                row.update(run_methods(inst, cfg, seeds, conv if j == 0 else None, tag))
                rows.append(add_gaps(row, opt, row["W"]))
            sub = pd.DataFrame(rows[-per_case:])
            print(f"  sim {label:10s} {row['case']:24s} "
                  + "  ".join(f"{k} {sub[k + ' gap%'].mean():5.2f}"
                              for k in ["WSPT", "best_rule"] + METHODS), flush=True)
    return pd.DataFrame(rows)


def exp_benchmark(cfg, seeds, per_family):
    from scheduling.suites import EXACT_SUITE, all_suites
    rows = []
    for fam, insts in all_suites().items():
        for inst in insts[:per_family]:
            exact = fam == EXACT_SUITE
            ref = exact_single_machine(inst) if exact else lower_bound(inst)
            row = dict(family=fam, instance=inst.name, n=inst.n, m=inst.m,
                       W=inst.total_weight, ref=ref, ref_kind="exact" if exact else "LP bound")
            row.update(baselines(inst))
            row.update(run_methods(inst, cfg, seeds))
            rows.append(add_gaps(row, ref, row["W"]))
        sub = pd.DataFrame(rows[-min(per_family, len(insts)):])
        print(f"  bench {fam:26s} "
              + "  ".join(f"{k} {sub[k + ' gap%'].mean():5.2f}"
                          for k in ["WSPT", "best_rule"] + METHODS), flush=True)
    return pd.DataFrame(rows)


def self_check():
    """The numba decoder must reproduce environment.decode_greedy_insert exactly,
    and decode_assign must never return an infeasible schedule."""
    from scheduling.environment import decode_greedy_insert
    from scheduling.instances import gen_p1
    rng = np.random.default_rng(7)
    for s in range(40):
        inst = gen_p1(n=30, m=int(rng.integers(1, 5)), seed=90000 + s,
                      tardiness_factor=float(rng.uniform(0.2, 0.8)))
        prio = rng.random(inst.n)
        ref, _ = decode_greedy_insert(inst, prio)
        order = np.argsort(-prio, kind="stable")
        kept, _ = decode_list(order, inst.p, inst.d, inst.w, inst.m)
        assert abs((inst.total_weight - kept) - ref) < 1e-6, "decode_list mismatch"
        kept, mach = decode_assign(order, rng.integers(0, inst.m, inst.n),
                                   inst.p, inst.d, inst.w, inst.m)
        for j in range(inst.m):           # each machine, run in EDD order, is on time
            idx = sorted(np.where(mach == j)[0], key=lambda i: inst.d[i])
            assert np.all(np.cumsum(inst.p[idx]) <= inst.d[idx] + 1e-6), "late job"
        assert abs(kept - inst.w[mach >= 0].sum()) < 1e-6
    print("  self-check passed (decoders verified on 40 random instances)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    todo = args or ["adversarial", "similarity", "benchmark"]
    os.makedirs(OUT, exist_ok=True)
    cfg = GAConfig(pop=60, gens=60, patience=40) if quick else GAConfig()
    seeds = [0] if quick else [0, 1, 2]
    t0 = time.time()
    self_check()
    conv, summary = [], []
    for exp in todo:
        print(f"\n=== {exp} ===", flush=True)
        if exp == "adversarial":
            df = exp_adversarial(cfg, seeds, conv)
        elif exp == "similarity":
            df = exp_similarity(cfg, seeds, conv, per_case=2 if quick else 5,
                                milp_limit=20 if quick else 60)
        elif exp == "benchmark":
            df = exp_benchmark(cfg, seeds, per_family=1 if quick else 2)
        else:
            raise SystemExit(f"unknown experiment {exp}")
        df.round(4).to_csv(os.path.join(OUT, f"{exp}.csv"), index=False)
        groups = ["setting"] if exp == "similarity" else [None]
        for gcol in groups:
            for key, sub in (df.groupby(gcol) if gcol else [(exp, df)]):
                r = dict(experiment=exp if gcol is None else f"{exp} {key}",
                         instances=len(sub))
                for k in ["WSPT", "best_rule"] + METHODS + ["GA-joint (best)"]:
                    r[k] = sub[k + " gap%"].mean()
                r["GA-joint beats WSPT on"] = int((sub["GA-joint"] < sub["WSPT"] - 1e-6).sum())
                r["GA-joint beats GA-priority on"] = int(
                    (sub["GA-joint"] < sub["GA-priority"] - 1e-6).sum())
                summary.append(r)
    S = pd.DataFrame(summary)
    S.round(3).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    if conv:
        pd.DataFrame(conv).to_csv(os.path.join(OUT, "convergence.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== mean % gap to the reference (exact optimum, LP bound on benchmark m>1) ===")
    print(S.round(2).to_string(index=False))
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
