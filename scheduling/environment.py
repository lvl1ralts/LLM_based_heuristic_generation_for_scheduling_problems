"""
Pm | r_i = 0 | sum_i w_i U_i  --  environment, decoders and bounds.

Scope note: this module is the Week-1 baseline environment only. The heuristic
never builds a schedule itself; it only returns a *priority score* per job and
the environment decides admission, so no heuristic can ever emit an infeasible
schedule.

Two decoders turn a priority vector into a schedule:

  greedy_insert : sort jobs by priority (desc); accept a job iff the accepted set
                  stays feasible under EDD list scheduling.
  mh_drop       : sweep jobs in EDD order, admit each; on infeasibility drop the
                  admitted job with the *lowest* priority.  With priority = w/p
                  this is exactly the weighted Moore-Hodgson heuristic (Brucker
                  Ch.4/Ch.5); with priority = 1/p and unit weights it is the
                  optimal Moore-Hodgson algorithm on a single machine.
"""
from __future__ import annotations

import heapq
from bisect import insort
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from scheduling.instances import P1Instance

EPS = 1e-9


# --------------------------------------------------------------------------- #
# feasibility oracle
# --------------------------------------------------------------------------- #
def edd_feasible(pd_sorted: Sequence[Tuple[float, float]], m: int) -> bool:
    """pd_sorted: list of (deadline, processing time) sorted by deadline ascending.

    Jackson's rule: a set of jobs with deadlines is feasible on m identical
    machines iff list-scheduling them in EDD order onto the earliest-free
    machine misses no deadline.
    """
    loads = [0.0] * m
    heapq.heapify(loads)
    for di, pi in pd_sorted:
        t = heapq.heappop(loads)
        c = t + pi
        if c > di + EPS:
            return False
        heapq.heappush(loads, c)
    return True


def schedule_of(pd_sorted, m):
    """Materialise the EDD list schedule: (job index, machine, start, finish)."""
    heap = [(0.0, j) for j in range(m)]
    heapq.heapify(heap)
    out = []
    for di, pi, idx in pd_sorted:
        t, j = heapq.heappop(heap)
        out.append((idx, j, t, t + pi))
        heapq.heappush(heap, (t + pi, j))
    return out


# --------------------------------------------------------------------------- #
# feature table handed to a heuristic
# --------------------------------------------------------------------------- #
def features(inst: P1Instance) -> Dict[str, np.ndarray]:
    """Per-job signals every priority rule is allowed to read."""
    key = "_p1feat"
    if key in inst.meta:
        return inst.meta[key]
    p, d, w, m, n = inst.p, inst.d, inst.w, inst.m, inst.n
    order = np.argsort(d, kind="stable")
    cum = np.zeros(n)
    cum[order] = np.cumsum(p[order])
    # Horn's condition slack: how much machine capacity is left before d_i once every
    # job with an earlier deadline is packed.  Negative => the EDD prefix is overloaded.
    headroom = m * d - cum
    f = dict(
        p=p,                                           # processing time
        d=d,                                           # deadline
        w=w,                                           # weight (value if on time)
        wp=w / np.maximum(p, EPS),                     # weight density (WSPT key)
        slack=d - p,                                   # absolute laxity
        laxity=(d - p) / np.maximum(d, EPS),           # relative laxity
        urgency=p / np.maximum(d, EPS),                # fraction of the window used
        dens=w / np.maximum(p * d, EPS),               # weight per unit of "resource-time"
        headroom=headroom,                             # Horn capacity left before d_i
        headroom_norm=headroom / max(float(p.sum()), EPS),
        d_rank=np.argsort(np.argsort(d)) / max(n - 1, 1),
        p_norm=p / max(float(p.max()), EPS),
        w_norm=w / max(float(w.max()), EPS),
        d_norm=d / max(float(d.max()), EPS),
        cum_p_before=cum,                              # work due no later than d_i
    )
    inst.meta[key] = f
    return f


def context(inst: P1Instance) -> Dict[str, float]:
    """Instance-level scalars every priority rule is allowed to read."""
    p, d = inst.p, inst.d
    return dict(
        m=float(inst.m),
        n=float(inst.n),
        total_p=float(p.sum()),
        total_w=float(inst.w.sum()),
        mean_p=float(p.mean()),
        max_d=float(d.max()),
        min_d=float(d.min()),
        mean_d=float(d.mean()),
        load=float(p.sum() / (inst.m * max(d.max(), EPS))),  # >1 => guaranteed rejections
    )


# --------------------------------------------------------------------------- #
# decoders
# --------------------------------------------------------------------------- #
def decode_greedy_insert(inst: P1Instance, prio: np.ndarray) -> Tuple[float, np.ndarray]:
    """Highest priority first; admit whenever the admitted set stays feasible."""
    p, d, w, m = inst.p, inst.d, inst.w, inst.m
    order = np.argsort(-prio, kind="stable")
    accepted: List[Tuple[float, float]] = []
    acc_mask = np.zeros(inst.n, dtype=bool)
    for i in order:
        if p[i] > d[i] + EPS:
            continue
        cand = list(accepted)
        insort(cand, (float(d[i]), float(p[i])))
        if edd_feasible(cand, m):
            accepted = cand
            acc_mask[i] = True
    return float(w[~acc_mask].sum()), acc_mask


def decode_mh_drop(inst: P1Instance, prio: np.ndarray) -> Tuple[float, np.ndarray]:
    """Sweep in deadline order, admit everything, evict lowest priority on overflow."""
    p, d, w, m = inst.p, inst.d, inst.w, inst.m
    order = np.argsort(d, kind="stable")
    acc: List[int] = []
    acc_mask = np.zeros(inst.n, dtype=bool)
    for i in order:
        if p[i] > d[i] + EPS:
            continue
        acc.append(int(i))
        acc_mask[i] = True
        while acc and not edd_feasible([(float(d[j]), float(p[j])) for j in acc], m):
            victim = min(acc, key=lambda j: (prio[j], -p[j]))
            acc.remove(victim)
            acc_mask[victim] = False
    return float(w[~acc_mask].sum()), acc_mask


DECODERS = {"greedy_insert": decode_greedy_insert, "mh_drop": decode_mh_drop}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(inst: P1Instance, prio_fn: Callable, decoder: str = "greedy_insert") -> float:
    """Objective value sum w_i U_i achieved by `prio_fn` under `decoder`."""
    prio = np.asarray(prio_fn(features(inst), context(inst)), dtype=np.float64).ravel()
    if prio.shape[0] != inst.n or not np.all(np.isfinite(prio)):
        prio = np.nan_to_num(
            np.resize(prio, inst.n), nan=0.0, posinf=1e12, neginf=-1e12
        )
    obj, _ = DECODERS[decoder](inst, prio)
    return obj


def accepted_mask(inst: P1Instance, prio_fn: Callable,
                  decoder: str = "greedy_insert") -> np.ndarray:
    prio = np.asarray(prio_fn(features(inst), context(inst)), dtype=np.float64).ravel()
    _, mask = DECODERS[decoder](inst, prio)
    return mask


def evaluate_suite(insts, prio_fn, decoder="greedy_insert") -> np.ndarray:
    return np.array([evaluate(i, prio_fn, decoder) for i in insts], dtype=float)


def normalised(insts, objs) -> np.ndarray:
    """(obj - LB) / (W_total - LB): 0 = matches the LP bound, 1 = reject everything."""
    out = []
    for inst, o in zip(insts, objs):
        lb = lower_bound(inst)
        span = max(inst.total_weight - lb, EPS)
        out.append(float(np.clip((o - lb) / span, 0.0, 2.0)))
    return np.array(out)


# --------------------------------------------------------------------------- #
# bounds
# --------------------------------------------------------------------------- #
def lower_bound(inst: P1Instance) -> float:
    """LP bound from Horn's preemptive feasibility condition (Brucker Ch.5).

        max sum_i w_i x_i   s.t.  sum_{i: d_i <= t} p_i x_i <= m t   for all t in {d_i}
                                  0 <= x <= 1
    Any non-preemptive feasible on-time set satisfies these constraints, so the LP
    optimum upper-bounds the achievable on-time weight and W - LP is a valid lower
    bound on sum w_i U_i.
    """
    key = "_lb"
    if key in inst.meta:
        return inst.meta[key]
    from scipy.optimize import linprog

    p, d, w, m, n = inst.p, inst.d, inst.w, inst.m, inst.n
    ts = np.unique(d)
    A = (d[None, :] <= ts[:, None]).astype(float) * p[None, :]
    b = m * ts
    res = linprog(
        c=-w, A_ub=A, b_ub=b, bounds=[(0.0, 1.0)] * n, method="highs"
    )
    on_time = -res.fun if res.success else float(w.sum())
    lb = max(0.0, float(w.sum()) - float(on_time))
    inst.meta[key] = lb
    return lb


def exact_single_machine(inst: P1Instance) -> float:
    """Exact DP for 1 | r_i = 0 | sum w_i U_i (Lawler-Moore).  O(n * sum p).

    Only valid for m = 1; used to measure a *true* optimality gap rather than a
    gap to the (possibly loose) LP relaxation.
    """
    assert inst.m == 1
    p = np.rint(inst.p).astype(int)
    d = np.rint(inst.d).astype(int)
    w = inst.w
    order = np.argsort(d, kind="stable")
    T = int(p.sum())
    NEG = -1e18
    dp = np.full(T + 1, NEG)
    dp[0] = 0.0
    for i in order:
        pi, di, wi = int(p[i]), int(d[i]), float(w[i])
        hi = min(di - pi, T - pi)
        if hi < 0:
            continue
        src = dp[: hi + 1]
        tgt = dp[pi : hi + pi + 1]
        np.maximum(tgt, src + wi, out=tgt)
    return float(inst.total_weight - dp.max())
