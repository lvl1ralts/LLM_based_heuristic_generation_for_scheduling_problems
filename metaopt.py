#!/usr/bin/env python3
"""
MetaOpt-style adversarial input search (Namyar et al., 2023) for the strongest
rules on  Pm | r_i = 0 | sum w_i U_i.

The bi-level problem

    max over I    OPT(I) - H(I)          (weight kept by the optimum minus
    s.t.  I in family's input set         weight kept by the heuristic)

is written as ONE mixed-integer program, following the paper's rewrites:

  * leader     : chooses the instance. Each job picks a "type" (p, w[, d]) from
                 the family's allowed values; the family's generator rules
                 (tardiness factor, due range, weight model, p model) become
                 linear constraints, so every instance found is one the
                 family's generator can produce.
  * heuristic  : a greedy algorithm, so it is encoded as a FEASIBILITY problem
                 (the paper's treatment of FFD): constraints force the
                 variables to reproduce exactly what greedy insertion does,
                 i.e. its priority order and, for every job, whether EDD list
                 scheduling of (admitted so far + that job) misses a deadline.
  * optimum    : a maximisation in the same direction as the leader, so it
                 merges with the leader: assignment of jobs to machines with
                 the exact per-machine deadline condition.

Every instance found is then re-run through the real code (scheduling/) and its
true optimum recomputed independently, which verifies the encoding.

Writes results/metaopt/<family>__<rule>.json and results/metaopt/summary.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduling.environment import (accepted_mask, decode_greedy_insert,
                                    edd_feasible, exact_single_machine)
from scheduling.heuristics import RULES
from scheduling.instances import P1Instance

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "metaopt")
os.makedirs(OUT, exist_ok=True)
EPS_PRIO = 0.0015   # strict priority margin: forbids ties, and exceeds the
                    # 0.001 tie-break of WSPT-EDD, so WSPT and WSPT-EDD share an order

# --------------------------------------------------------------------------- #
# families: scaled-down versions of the 12 benchmark families
# --------------------------------------------------------------------------- #
P_GRID = {
    "uniform":   [5, 20, 40, 60, 80, 100],
    "bimodal":   [1, 4, 8, 11, 70, 85, 100],
    "lognormal": [3, 8, 20, 50, 120, 300],
}
W_UNCORR = [1, 3, 5, 7, 10]


def weight_options(p, model):
    if model == "uncorrelated":
        return W_UNCORR
    base = round(p / 10.0) if model == "correlated" else round((100 - p) / 10.0)
    return sorted({base + 1, base + 4})          # generator: base + U{0..3} + 1


FAMILIES = {
    # name            n  m   T     R    p model      weight model
    # n is limited by what the exact MILP can solve: 6 jobs on 2 machines,
    # 8 on 1 machine (the paper scales the same way, or partitions).
    "in-dist":       (6, 2, 0.4, 0.6, "uniform",   "uncorrelated"),
    "scale-shift":   (8, 2, 0.5, 0.6, "uniform",   "uncorrelated"),
    "bimodal p":     (6, 2, 0.5, 0.6, "bimodal",   "uncorrelated"),
    "lognormal p":   (6, 2, 0.5, 0.6, "lognormal", "uncorrelated"),
    "tight T=0.8":   (8, 1, 0.8, 0.3, "uniform",   "uncorrelated"),  # 2 machines leave no room for distinct deadlines
    "loose T=0.1":   (6, 2, 0.1, 0.6, "uniform",   "uncorrelated"),
    "w correlated":  (6, 2, 0.5, 0.6, "uniform",   "correlated"),
    "w anticorr":    (6, 2, 0.5, 0.6, "uniform",   "anticorrelated"),
    "narrow R=0.1":  (6, 2, 0.5, 0.1, "uniform",   "uncorrelated"),
    "many machines": (6, 3, 0.5, 0.6, "uniform",   "uncorrelated"),
    "single m=1":    (8, 1, 0.5, 0.6, "uniform",   "uncorrelated"),
    "large n":       (10, 1, 0.5, 0.6, "uniform",  "uncorrelated"),
}

# rule -> (priority as a function of (p, w, d), whether it depends on d)
PRIORITY = {
    "WSPT":        (lambda p, w, d: w / p, False),
    "WSPT-Laxity": (lambda p, w, d: (w / p) * (1.0 + (d - p) / d), True),
}


# --------------------------------------------------------------------------- #
# a small sparse MILP builder
# --------------------------------------------------------------------------- #
class Model:
    def __init__(self):
        self.lo, self.hi, self.integ, self.cost = [], [], [], []
        self.rows, self.cols, self.vals, self.rlo, self.rhi = [], [], [], [], []

    def var(self, lo=0.0, hi=1.0, integer=True):
        self.lo.append(lo); self.hi.append(hi)
        self.integ.append(1 if integer else 0); self.cost.append(0.0)
        return len(self.lo) - 1

    def add(self, expr, lo=-np.inf, hi=np.inf):
        """expr: (dict var->coef, constant). Adds lo <= expr <= hi."""
        terms, const = expr
        r = len(self.rlo)
        for v, c in terms.items():
            if c != 0:
                self.rows.append(r); self.cols.append(v); self.vals.append(c)
        self.rlo.append(lo - const); self.rhi.append(hi - const)

    def prod(self, expr, X, b):
        """z = expr * b, expr in [0, X], b binary (McCormick, exact)."""
        z = self.var(0, X, integer=False)
        self.add(lin({z: 1, b: -X}), hi=0)                         # z <= X b
        self.add(add(lin({z: 1}), scale(expr, -1)), hi=0)          # z <= expr
        self.add(add(add(lin({z: 1}), scale(expr, -1)), lin({b: -X})), lo=-X)  # z >= expr - X(1-b)
        return z

    def solve(self, time_limit):
        n = len(self.lo)
        A = coo_matrix((self.vals, (self.rows, self.cols)), shape=(len(self.rlo), n)).tocsr()
        return milp(np.array(self.cost), constraints=LinearConstraint(A, self.rlo, self.rhi),
                    integrality=np.array(self.integ), bounds=Bounds(self.lo, self.hi),
                    options=dict(time_limit=time_limit, disp=False))


def lin(d, c=0.0):
    return (dict(d), c)


def add(a, b):
    t = dict(a[0])
    for k, v in b[0].items():
        t[k] = t.get(k, 0.0) + v
    return (t, a[1] + b[1])


def scale(a, s):
    return ({k: v * s for k, v in a[0].items()}, a[1] * s)


# --------------------------------------------------------------------------- #
def build(family, rule, d_levels=16):
    n, m, T, R, pmodel, wmodel = FAMILIES[family]
    prio_fn, needs_d = PRIORITY[rule]
    pg = P_GRID[pmodel]
    pmax, wmax = max(pg), max(max(weight_options(p, wmodel)) for p in pg)
    lo_c, hi_c = (1 - T - R / 2) / m, (1 - T + R / 2) / m
    dmax = int(max(hi_c * n * pmax, pmax) + 2)
    types = [(p, w, None) for p in pg for w in weight_options(p, wmodel)]
    if needs_d:
        # WSPT-Laxity: (w/p)(1 + (d-p)/d) = 2 w/p - w/d.  d is chosen from a grid
        # of levels; 1/d is then linear in the level choice, and w * (1/d) is an
        # exact McCormick product of a binary type choice with a bounded term.
        dg = sorted({int(round(x)) for x in np.linspace(min(pg), dmax, d_levels)})
        allp = [2 * w / p - w / d for p, w, _ in types for d in dg if d >= p]
        prio = [2 * w / p for p, w, _ in types]
        Mp = max(allp) - min(allp) + 2 * EPS_PRIO
    else:
        prio = [prio_fn(p, w, 1.0) for p, w, _ in types]
        Mp = max(prio) - min(prio) + 2 * EPS_PRIO
    Lmax = float(n * pmax)
    M = Lmax + dmax + 1

    md = Model()
    # ---- leader: instance -------------------------------------------------- #
    u = [[md.var() for _ in types] for _ in range(n)]
    for i in range(n):
        md.add(lin({v: 1 for v in u[i]}), 1, 1)
    P = [lin({u[i][t]: types[t][0] for t in range(len(types))}) for i in range(n)]
    W = [lin({u[i][t]: types[t][1] for t in range(len(types))}) for i in range(n)]
    PR = [lin({u[i][t]: prio[t] for t in range(len(types))}) for i in range(n)]
    if needs_d:
        D = []
        for i in range(n):
            z = [md.var() for _ in dg]
            md.add(lin({v: 1 for v in z}), 1, 1)
            D.append(lin({z[k]: dg[k] for k in range(len(dg))}))
            invd = lin({z[k]: 1.0 / dg[k] for k in range(len(dg))})
            for t in range(len(types)):         # priority -= w_t * (1/d) * u_t
                q = md.prod(invd, 1.0 / dg[0], u[i][t])
                PR[i] = add(PR[i], lin({q: -types[t][1]}))
    else:
        D = [lin({md.var(1, dmax, integer=True): 1}) for _ in range(n)]
    for i in range(n - 1):                                   # jobs indexed in deadline order
        md.add(add(D[i + 1], scale(D[i], -1)), lo=1)
    # generator window: lo = max(lo_c*sum p, max p), hi = max(hi_c*sum p, lo+1)
    SP = lin({})
    for i in range(n):
        SP = add(SP, P[i])
    pstar = md.var(0, pmax, integer=False)
    mu = [md.var() for _ in range(n)]
    md.add(lin({x: 1 for x in mu}), 1, 1)
    for i in range(n):
        md.add(add(lin({pstar: 1}), scale(P[i], -1)), lo=0)
        md.add(add(add(lin({pstar: 1}), scale(P[i], -1)), lin({mu[i]: pmax})), hi=pmax)
    LO = md.var(-M, M, integer=False)
    lam = md.var()
    md.add(add(lin({LO: 1}), scale(SP, -lo_c)), lo=0)
    md.add(lin({LO: 1, pstar: -1}), lo=0)
    md.add(add(lin({LO: 1, lam: -M}), scale(SP, -lo_c)), hi=0)
    md.add(lin({LO: 1, pstar: -1, lam: M}), hi=M)
    UP = md.var(-M, M, integer=False)
    kap = md.var()
    md.add(add(lin({UP: 1}), scale(SP, -hi_c)), lo=0)
    md.add(lin({UP: 1, LO: -1}), lo=1)
    md.add(add(lin({UP: 1, kap: -M}), scale(SP, -hi_c)), hi=0)
    md.add(lin({UP: 1, LO: -1, kap: M}), hi=1 + M)
    for i in range(n):
        md.add(add(D[i], lin({LO: -1})), lo=0)
        md.add(add(D[i], lin({UP: -1})), hi=0)

    # ---- heuristic: priority order ---------------------------------------- #
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            y[i, j] = md.var()
            md.add(add(add(PR[i], scale(PR[j], -1)), lin({y[i, j]: -Mp})), lo=EPS_PRIO - Mp)
            md.add(add(add(PR[j], scale(PR[i], -1)), lin({y[i, j]: Mp})), lo=EPS_PRIO)

    def before(k, i):                     # expr: 1 if job k is ranked before job i
        return lin({y[k, i]: 1}) if k < i else lin({y[i, k]: -1}, 1.0)

    a = [md.var() for _ in range(n)]       # heuristic admits job i
    # ---- heuristic: one EDD list-scheduling simulation per admission decision
    for i in range(n):
        s = []
        for k in range(n):
            if k == i:
                s.append(lin({}, 1.0))
                continue
            b = md.var()
            bk = before(k, i)
            md.add(add(lin({b: 1}), scale(bk, -1)), hi=0)
            md.add(lin({b: 1, a[k]: -1}), hi=0)
            md.add(add(add(lin({b: 1}), scale(bk, -1)), lin({a[k]: -1})), lo=-1)
            s.append(lin({b: 1}))
        loads = [lin({}) for _ in range(m)]
        viol = []
        for k in range(n):
            g = [md.var() for _ in range(m)] if m > 1 else None
            if m == 1:
                start = loads[0]
                e = md.prod(P[k], pmax, list(s[k][0].keys())[0]) if k != i else None
                new0 = add(loads[0], lin({e: 1}) if e is not None else P[k])
                loads = [lin(new0[0], new0[1])]
                if k != i:
                    L = md.var(0, Lmax, integer=False)
                    md.add(add(lin({L: 1}), scale(loads[0], -1)), 0, 0)
                    loads = [lin({L: 1})]
            else:
                md.add(add(lin({v: 1 for v in g}), scale(s[k], -1)), 0, 0)
                for j in range(m):              # g_j = 1 only on a least-loaded machine
                    for j2 in range(m):
                        if j2 != j:
                            md.add(add(add(loads[j], scale(loads[j2], -1)), lin({g[j]: M})), hi=M)
                h = [md.prod(loads[j], Lmax, g[j]) for j in range(m)]
                start = lin({x: 1 for x in h})
                newl = []
                for j in range(m):
                    e = md.prod(P[k], pmax, g[j])
                    L = md.var(0, Lmax, integer=False)
                    md.add(add(add(lin({L: 1}), scale(loads[j], -1)), lin({e: -1})), 0, 0)
                    newl.append(lin({L: 1}))
                loads = newl
            if k < i:
                continue                         # earlier jobs are unaffected by adding i
            C = add(start, P[k])
            sk_const = (k == i)
            # admitted -> no violation:  C - d_k <= M(1-s) + M(1-a_i)
            if sk_const:
                md.add(add(add(C, scale(D[k], -1)), lin({a[i]: M})), hi=M)
            else:
                md.add(add(add(add(C, scale(D[k], -1)), scale(s[k], M)), lin({a[i]: M})), hi=2 * M)
            # rejected -> some witnessed violation: C - d_k >= 1 - M(1-v)
            v = md.var()
            if not sk_const:
                md.add(add(lin({v: 1}), scale(s[k], -1)), hi=0)
            md.add(add(add(C, scale(D[k], -1)), lin({v: -M})), lo=1 - M)
            viol.append(v)
        md.add(lin({**{v: 1 for v in viol}, a[i]: 1}), lo=1)

    # ---- optimum: per-machine EDD condition -------------------------------- #
    x = [[md.var() for _ in range(m)] for _ in range(n)]
    for k in range(n):
        md.add(lin({x[k][j]: 1 for j in range(m)}), hi=1)
    r = [[md.prod(P[l], pmax, x[l][j]) for j in range(m)] for l in range(n)]
    for j in range(m):
        for k in range(n):
            md.add(add(lin({**{r[l][j]: 1 for l in range(k + 1)}, x[k][j]: M}), scale(D[k], -1)), hi=M)
    for k in range(n):                           # break machine symmetry: job k uses
        for j in range(k + 1, m):                # only machines 0..k (machines are identical)
            md.add(lin({x[k][j]: 1}), hi=0)

    # ---- objective: max  sum w*o  -  sum w*a ------------------------------ #
    for k in range(n):
        za = md.prod(W[k], wmax, a[k])
        md.cost[za] += 1.0
        for j in range(m):
            zo = md.prod(W[k], wmax, x[k][j])
            md.cost[zo] -= 1.0
    return md, dict(n=n, m=m, types=types, u=u, a=a, x=x, D=D, needs_d=needs_d)


def extract(res, meta):
    xs = res.x
    n, m, types = meta["n"], meta["m"], meta["types"]
    p, w, d = [], [], []
    for i in range(n):
        t = int(np.argmax([xs[v] for v in meta["u"][i]]))
        p.append(types[t][0]); w.append(types[t][1])
        dexpr = meta["D"][i]
        d.append(round(sum(xs[v] * c for v, c in dexpr[0].items()) + dexpr[1]))
    a = [xs[v] > 0.5 for v in meta["a"]]
    o = [sum(xs[meta["x"][k][j]] for j in range(m)) > 0.5 for k in range(n)]
    return np.array(p, float), np.array(d, float), np.array(w, float), np.array(a), np.array(o)


# --------------------------------------------------------------------------- #
def exact_opt(inst, time_limit=120):
    """True optimum and an optimal assignment, solved independently of the
    encoding above. For m = 1 the value is cross-checked against the DP."""
    from scipy.sparse import lil_matrix
    p, d, w, m, n = inst.p, inst.d, inst.w, inst.m, inst.n
    ts = np.unique(d)
    nv = n * m + n
    c = np.zeros(nv); c[n * m:] = -w
    A = lil_matrix((n + m * len(ts), nv)); lo, hi = [], []
    for i in range(n):
        for j in range(m):
            A[i, i * m + j] = 1
        A[i, n * m + i] = -1; lo.append(0); hi.append(0)
    r = n
    for j in range(m):
        for t in ts:
            for i in np.where(d <= t)[0]:
                A[r, i * m + j] = p[i]
            lo.append(-np.inf); hi.append(t); r += 1
    # With scipy 1.18.1 (HiGHS), presolve returned a "proved optimal" solution 7 below
    # the true optimum on a 40-job, 4-machine instance (sim_s31303), found when the GA
    # beat it. Without presolve the proofs are sound but slower, so it can stop at the
    # time limit with a worse schedule than a presolve run finds. Hence: solve without
    # presolve; if that is not proved, also solve with presolve; keep the better
    # schedule, and call it proved only when the presolve-free run proved it.
    cons = LinearConstraint(A.tocsr(), lo, hi)
    value, assign, proved = np.inf, {}, False
    for presolve in (False, True):
        opts = dict(time_limit=time_limit, mip_rel_gap=0, presolve=presolve)
        res = milp(c, constraints=cons, integrality=np.ones(nv), bounds=Bounds(0, 1),
                   options=opts)
        if res.x is None:
            continue
        keep = res.x[n * m:] > 0.5
        asg = {int(i): int(np.argmax(res.x[i * m:(i + 1) * m])) for i in np.where(keep)[0]}
        for j in range(m):                     # every machine on time in EDD order
            idx = sorted((i for i, mj in asg.items() if mj == j), key=lambda i: d[i])
            assert np.all(np.cumsum(p[idx]) <= d[idx] + 1e-6), "MILP returned a late job"
        v = float(inst.total_weight - w[keep].sum())
        if not presolve and res.status == 0:
            value, assign, proved = v, asg, True
            break
        if v < value:
            value, assign = v, asg
    if m == 1:
        assert not proved or abs(value - exact_single_machine(inst)) < 1e-6, \
            "MILP and DP disagree"
    return value, assign, proved


def analyse(family, rule, res, meta, solve_s):
    p, d, w, a_enc, o_enc = extract(res, meta)
    m = meta["m"]
    inst = P1Instance(p=p, d=d, w=w, m=m, name=f"{family}|{rule}")
    W = float(w.sum())
    opt, opt_assign, opt_proved = exact_opt(inst)
    opt_keep = np.zeros(inst.n, bool)
    opt_keep[list(opt_assign)] = True
    out = dict(family=family, encoded_rule=rule, n=inst.n, m=m,
               p=p.tolist(), d=d.tolist(), w=w.tolist(), total_w=W,
               solver_status=int(res.status), solver_proved=bool(res.status == 0),
               solve_seconds=round(solve_s, 1), opt_lost=opt, opt_proved=bool(opt_proved),
               opt_keep=opt_keep.tolist(), opt_assign=opt_assign)
    rules = {}
    for rn in ["WSPT", "WSPT-EDD", "WSPT-Laxity"]:
        keep = accepted_mask(inst, RULES[rn].fn, RULES[rn].decoder)
        lost = float(w[~keep].sum())
        rules[rn] = dict(lost=lost, keep=keep.tolist(),
                         gap_pct=100.0 * (lost - opt) / max(W - opt, 1e-9),
                         ratio=(lost / opt) if opt > 0 else (float("inf") if lost > 0 else 1.0))
    out["rules"] = rules
    enc_lost = float(w[~a_enc].sum())
    # the encoded heuristic must reproduce the real code exactly
    out["encoding_verified"] = bool(abs(enc_lost - rules[rule]["lost"]) < 1e-6)
    # the encoded optimum equals the independent one when the search finished
    out["encoded_opt_is_optimal"] = bool(abs((W - float(w[o_enc].sum())) - opt) < 1e-6)
    out["encoded_heuristic_lost"] = enc_lost
    out["encoded_opt_lost"] = W - float(w[o_enc].sum())
    # cause: can any priority rule reach the optimal set through this decoder?
    S = sorted((float(d[i]), float(p[i])) for i in np.where(opt_keep)[0])
    out["opt_set_passes_decoder_test"] = bool(edd_feasible(S, m))
    out["cause"] = "ranking" if out["opt_set_passes_decoder_test"] else "decoder"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--rules", default="WSPT,WSPT-Laxity")
    ap.add_argument("--time", type=float, default=240)
    ap.add_argument("--dlevels", type=int, default=16,
                    help="deadline grid size for WSPT-Laxity (finer = slower)")
    args = ap.parse_args()
    for fam in args.families.split(","):
        for rule in args.rules.split(","):
            t0 = time.time()
            md, meta = build(fam, rule, d_levels=args.dlevels)
            res = md.solve(args.time)
            dt = time.time() - t0
            if res.x is None:
                print(f"{fam:14s} {rule:12s} no solution in {dt:.0f}s", flush=True)
                continue
            out = analyse(fam, rule, res, meta, dt)
            fn = f"{fam.replace(' ', '_').replace('=', '')}__{rule}.json"
            json.dump(out, open(os.path.join(OUT, fn), "w"), indent=1)
            g = out["rules"][rule]
            print(f"{fam:14s} {rule:12s} gap {g['gap_pct']:6.2f}%  lost {g['lost']:.0f} vs opt "
                  f"{out['opt_lost']:.0f}  cause={out['cause']:8s} verified={out['encoding_verified']} "
                  f"proved={out['solver_proved']}  ({dt:.0f}s, {len(md.lo)} vars)", flush=True)


if __name__ == "__main__":
    main()
