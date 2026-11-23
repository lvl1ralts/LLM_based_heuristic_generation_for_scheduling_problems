#!/usr/bin/env python3
"""
Figures for the GA report, from the CSVs that ga.py writes.

  python report_ga.py

Writes to results/ga/figures/:
  mechanism.png     adversarial instances: mean gap per method, ranking vs decoder failures
  cases_m4.png      the eight similarity cases on four machines
  convergence.png   best-so-far gap per generation, mean over instances
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "ga")
FIG = os.path.join(SRC, "figures")
os.makedirs(FIG, exist_ok=True)
EPS = 1e-9

COLOR = {"WSPT": "#a5a5a5", "best_rule": "#d4d4d4", "GA-priority": "#5b9bd5",
         "GA-assignment": "#70ad47", "GA-joint": "#1f4e79"}
LABEL = {"WSPT": "WSPT", "best_rule": "best of 17 rules", "GA-priority": "GA-priority",
         "GA-assignment": "GA-assignment", "GA-joint": "GA-joint"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["IBM Plex Sans", "Segoe UI", "Arial"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#e6e6e6", "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 200,
})


def bars(ax, groups, data, methods, cap=None):
    """Grouped bars; values above `cap` are cut and labelled."""
    k = len(methods)
    width = 0.8 / k
    x = np.arange(len(groups))
    for i, m in enumerate(methods):
        v = np.array(data[m], float)
        shown = np.minimum(v, cap) if cap else v
        ax.bar(x + (i - (k - 1) / 2) * width, shown, width, color=COLOR[m], label=LABEL[m])
        for xi, vi in zip(x, v):
            if cap and vi > cap:
                ax.text(xi + (i - (k - 1) / 2) * width, cap + cap * 0.02, f"{vi:.0f}",
                        ha="center", va="bottom", fontsize=7, color="#c55a11")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("mean % gap")


def mechanism():
    a = pd.read_csv(os.path.join(SRC, "adversarial.csv"))
    a = a[a["WSPT"] > a["opt"] + EPS]           # the 21 instances where WSPT is not optimal
    methods = ["WSPT", "best_rule", "GA-priority", "GA-assignment", "GA-joint"]
    groups, data = [], {m: [] for m in methods}
    for cause, name in [("ranking", "ranking failures"), ("decoder", "decoder failures")]:
        sub = a[a.cause == cause]
        groups.append(f"{name} ({len(sub)})")
        for m in methods:
            data[m].append(sub[m + " gap%"].mean())
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    bars(ax, groups, data, methods)
    for i, m in enumerate(methods):           # mark the zeros so they are visible
        for xi, v in enumerate(data[m]):
            if v < 0.05:
                ax.text(xi + (i - 2) * 0.16, 0.4, "0", ha="center", fontsize=7,
                        color=COLOR[m])
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mechanism.png"))
    plt.close(fig)


def cases_m4():
    s = pd.read_csv(os.path.join(SRC, "similarity.csv"))
    s = s[s.m == 4]
    order = list(dict.fromkeys(s["case"]))
    short = [c.replace(" similar", "").replace(" & ", ",").replace("none", "none")
             for c in order]
    methods = ["WSPT", "GA-priority", "GA-assignment", "GA-joint"]
    g = s.groupby("case")
    data = {m: [g.get_group(c)[m + " gap%"].mean() for c in order] for m in methods}
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    bars(ax, short, data, methods)
    ax.set_xlabel("similar numbers")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cases_m4.png"))
    plt.close(fig)


G = 300
LARGE = ["in-dist n50-100", "shift: very tight T=0.8", "weights correlated",
         "many machines m=16"]


def _pad(h):
    """A run that stopped early keeps its final value to generation G."""
    h = np.asarray(h, float)
    return np.concatenate([h, np.full(G - len(h), h[-1])])


def large_runs():
    """Fresh GA runs on one larger instance per family in LARGE (seed 0,
    patience off so every curve has G generations), plus the joint GA without
    the rule seeds. Gap to the LP bound."""
    from dataclasses import replace
    from ga import GA, GAConfig, MODE
    from scheduling.environment import lower_bound
    from scheduling.suites import all_suites
    S = all_suites()
    insts = [next(i for i in S[f] if i.n >= 100) for f in LARGE]
    runs = {m: MODE[m] for m in ["GA-priority", "GA-assignment", "GA-joint"]}
    runs["GA-joint, unseeded"] = "joint"
    out = {k: [] for k in runs}
    for inst in insts:
        lb, W = lower_bound(inst), inst.total_weight
        for k, mode in runs.items():
            cfg = replace(GAConfig(), mode=mode, patience=G, seeded="unseeded" not in k)
            h = GA(inst, cfg).run()["history"]
            out[k].append(100 * (_pad(h) - lb) / max(W - lb, EPS))
    return {k: np.mean(v, axis=0) for k, v in out.items()}, insts


def convergence():
    c = pd.read_csv(os.path.join(SRC, "convergence.csv"))
    s = pd.read_csv(os.path.join(SRC, "similarity.csv"))
    ref = {}
    for (label, case), sub in s.groupby(["setting", "case"]):
        r = sub.iloc[0]                      # convergence is logged for the first instance
        ref[f"{label}|{case}|0"] = (r.opt, r.W)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    for m in ["GA-priority", "GA-assignment", "GA-joint"]:
        curves = []
        for inst, sub in c[(c.method == m) & c.instance.str.startswith("m=4")].groupby("instance"):
            opt, W = ref[inst]
            h = sub.sort_values("gen")["lost"].to_numpy()
            curves.append(100 * (_pad(h) - opt) / max(W - opt, EPS))
        ax.plot(np.arange(G), np.mean(curves, axis=0), color=COLOR[m], lw=1.6, label=m)
    ax.set_title("similarity, m=4, n=40 (8 instances)\ngap to the exact optimum",
                 fontsize=8.5, loc="left")
    large, insts = large_runs()
    ax = axes[1]
    for k, v in large.items():
        base = k.split(",")[0]
        ax.plot(np.arange(G), v, color=COLOR[base], lw=1.6, ls="--" if "unseeded" in k else "-",
                label=k)
    ax.set_title(f"{len(insts)} benchmark instances, n=100-200, m=4-16\ngap to the LP bound",
                 fontsize=8.5, loc="left")
    for ax in axes:
        ax.set_xlabel("generation")
        ax.set_ylabel("mean % gap (best so far)")
        ax.set_xlim(0, G - 1)
        ax.set_ylim(bottom=0)
    axes[1].legend(fontsize=7.5)
    for k, v in large.items():
        print(f"  large instances, {k:20s} gen 0 {v[0]:6.2f}  gen 50 {v[50]:6.2f}  "
              f"gen {G - 1} {v[-1]:6.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "convergence.png"))
    plt.close(fig)


if __name__ == "__main__":
    mechanism()
    cases_m4()
    convergence()
    print(f"figures -> {FIG}")
