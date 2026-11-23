#!/usr/bin/env python3
"""
Figures for the phase-one presentation (../btp_presentation/), the same graphs
and the same values as the IEEE report (report_paper.py), redrawn for a 16:9
slide: every graph has a title, text is 14 pt or larger at its size on the
slide, and only major gridlines are drawn.

  python report_slides.py

Writes PNGs (300 dpi) to ../btp_presentation/figures/.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from report_paper import (ADV_LABEL, ADV_WORST, BLUE, EPS, GREEN, GREY, LIGHT, NAVY, ORANGE, RES,
                          SIM_CASES, SIM_M4)
from scheduling.environment import accepted_mask, schedule_of
from scheduling.heuristics import RULES
from scheduling.instances import P1Instance

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "btp_presentation", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 17, "axes.titlesize": 19, "axes.titleweight": "bold",
    "axes.labelsize": 17, "legend.fontsize": 16, "xtick.labelsize": 16,
    "ytick.labelsize": 16, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e2e2e2", "axes.axisbelow": True,
    "legend.frameon": False,
})
SIDE = (7.3, 5.4)        # graph beside the text
WIDE = (12.1, 4.7)       # graph across the slide


def save(fig, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)


def rules_benchmark():
    df = pd.read_csv(os.path.join(RES, "benchmark_overall.csv")).sort_values("overall")
    colors = [NAVY if r.startswith("WSPT") else ORANGE if r == "Random"
              else BLUE if r.startswith("M") and r not in ("MaxWeight", "MinSlack")
              else GREY for r in df.rule]
    fig, ax = plt.subplots(figsize=SIDE)
    y = np.arange(len(df))[::-1]
    ax.barh(y, df.overall, color=colors, height=0.72)
    for yi, v in zip(y, df.overall):
        ax.text(v + 0.6, yi, f"{v:.2f}", va="center", fontsize=15)
    ax.set_yticks(y)
    ax.set_yticklabels(df.rule, fontsize=15)
    ax.set_xlabel("mean gap to the LP bound (%)")
    ax.set_xlim(0, df.overall.max() * 1.17)
    ax.grid(axis="y", visible=False)
    ax.set_title("17 priority rules on 12 instance families", loc="left")
    ax.legend(handles=[Patch(color=NAVY, label="WSPT family"),
                       Patch(color=BLUE, label="evict decoder"),
                       Patch(color=ORANGE, label="random")], loc="center right")
    save(fig, "rules_benchmark")


def adversarial_worst():
    fig, ax = plt.subplots(figsize=SIDE)
    y = np.arange(len(ADV_WORST))[::-1]
    col = {"ranking": BLUE, "decoder": NAVY}
    for yi, (name, g, cause) in zip(y, ADV_WORST):
        ax.barh(yi, g, height=0.72, color=col[cause.split(",")[0]])
        ax.text(g + 0.8, yi, f"{g:.1f}", va="center", fontsize=15)
    ax.set_yticks(y)
    ax.set_yticklabels([ADV_LABEL[f[0]] for f in ADV_WORST])
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("worst gap to the optimum (%)")
    ax.set_xlim(0, 70)
    ax.grid(axis="y", visible=False)
    ax.set_title("Worst gap MetaOpt found", loc="left")
    ax.legend(handles=[Patch(color=NAVY, label="mainly decoder failure"),
                       Patch(color=BLUE, label="mainly ranking failure")],
              loc="lower right", bbox_to_anchor=(1.02, 0.0), fontsize=13,
              handlelength=1.2)
    save(fig, "adversarial_worst")


def decoder_failure(name="loose_T0.1__WSPT"):
    J = json.load(open(os.path.join(RES, "metaopt", name + ".json")))
    inst = P1Instance(p=J["p"], d=J["d"], w=J["w"], m=J["m"])
    keep = accepted_mask(inst, RULES["WSPT"].fn, RULES["WSPT"].decoder)
    rows = sorted((inst.d[i], inst.p[i], i) for i in np.where(keep)[0])
    wspt = list(schedule_of(rows, inst.m))
    opt = []
    for j in range(inst.m):
        t = 0.0
        for i in sorted((int(k) for k, v in J["opt_assign"].items() if v == j),
                        key=lambda i: inst.d[i]):
            opt.append((i, j, t, t + inst.p[i]))
            t += inst.p[i]
    fig, axes = plt.subplots(2, 1, figsize=(7.3, 5.0), sharex=True)
    titles = [f"WSPT: J3 is rejected, {J['rules']['WSPT']['lost']:.0f} lost",
              f"Optimum: all six jobs on time, {J['opt_lost']:.0f} lost"]
    kept = {x[0] for x in wspt}
    for ax, sched, title in zip(axes, [wspt, opt], titles):
        for i, j, s, e in sched:
            ax.barh(j, e - s, left=s, height=0.62, edgecolor="white",
                    color=BLUE if i not in kept else LIGHT)
            ax.text((s + e) / 2, j, f"J{i + 1}", ha="center", va="center", fontsize=16)
        ax.axvline(inst.d[2], color=ORANGE, ls="--", lw=2)
        ax.set_yticks(range(inst.m))
        ax.set_yticklabels([f"M{j + 1}" for j in range(inst.m)])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=18)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("time (dashed line: deadline of J3)")
    save(fig, "decoder_failure")


def similarity_m4():
    cols = [NAVY, BLUE, GREEN, GREY, ORANGE]
    g = pd.DataFrame(SIM_M4, index=SIM_CASES).T
    cap, width, x = 7.0, 0.16, np.arange(len(SIM_CASES))
    fig, ax = plt.subplots(figsize=WIDE)
    for k, (r, col) in enumerate(zip(SIM_M4, cols)):
        v = g.loc[r].to_numpy()
        xs = x + (k - 2) * width
        ax.bar(xs, np.minimum(v, cap), width, color=col, label=r)
        for xi, vi in zip(xs, v):
            if vi > cap:
                ax.text(xi, cap + 0.12, f"{vi:.0f}", ha="center", va="bottom",
                        fontsize=16, color=ORANGE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" similar", "").replace(" & ", ", ") for c in SIM_CASES])
    ax.set_xlabel("quantities that are similar")
    ax.set_ylabel("mean gap to optimum (%)")
    ax.set_ylim(0, cap + 1.0)
    ax.grid(axis="x", visible=False)
    ax.set_title("Four machines, 40 jobs: the five leading rules in the eight cases",
                 loc="left", pad=44)
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2,
              columnspacing=1.2, handlelength=1.3)
    save(fig, "similarity_m4")


def ga_ablation():
    a = pd.read_csv(os.path.join(RES, "ga", "adversarial.csv"))
    a = a[a["WSPT"] > a["opt"] + EPS]
    meths = [("WSPT", GREY), ("best_rule", LIGHT), ("GA-priority", BLUE),
             ("GA-assignment", GREEN), ("GA-joint", NAVY)]
    labels = {"best_rule": "best of 17 rules"}
    groups = [("ranking", "ranking failures"), ("decoder", "decoder failures")]
    width, x = 0.16, np.arange(2)
    fig, ax = plt.subplots(figsize=(7.3, 5.0))
    for k, (m, col) in enumerate(meths):
        v = [a[a.cause == c][m + " gap%"].mean() for c, _ in groups]
        xs = x + (k - 2) * width
        ax.bar(xs, v, width, color=col, label=labels.get(m, m))
        for xi, vi in zip(xs, v):
            lab = "0" if vi < 0.05 else f"{vi:.1f}" if vi < 1 else f"{vi:.0f}"
            ax.text(xi, vi + 0.5, lab, ha="center", fontsize=15)
    n = [int((a.cause == c).sum()) for c, _ in groups]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} ({k} instances)" for (_, lab), k in zip(groups, n)])
    ax.set_ylabel("mean gap to optimum (%)")
    ax.set_ylim(0, 30)
    ax.grid(axis="x", visible=False)
    ax.set_title("GA versions on the instances where WSPT fails", loc="left", pad=74)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2,
              columnspacing=1.0, handlelength=1.5)
    save(fig, "ga_ablation")


if __name__ == "__main__":
    rules_benchmark()
    adversarial_worst()
    decoder_failure()
    similarity_m4()
    ga_ablation()
    print(f"figures -> {os.path.abspath(OUT)}")
