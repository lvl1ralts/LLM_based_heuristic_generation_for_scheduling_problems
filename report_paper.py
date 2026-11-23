#!/usr/bin/env python3
"""
Figures for the IEEE-format phase-one report (../btp_report_phase1/), sized for
a two-column page (single column 3.5 in, full width 7.16 in). Every value
plotted is the one printed in the weekly reports.

  python report_paper.py

Writes PDFs to ../btp_report_phase1/figures/:
  rules_benchmark.pdf     all 17 rules, mean gap over the 12 families (report 01)
  adversarial_worst.pdf   worst gap found per family with MetaOpt (report 05)
  decoder_failure.pdf     WSPT's schedule vs the optimum, loose T=0.1 (report 04)
  similarity_m4.pdf       leading rules on the eight similarity cases (report 09)
  ga_ablation.pdf         GA variants on ranking vs decoder failures (report 10)
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scheduling.environment import accepted_mask, schedule_of
from scheduling.heuristics import RULES
from scheduling.instances import P1Instance

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "btp_report_phase1", "figures")
os.makedirs(OUT, exist_ok=True)
EPS = 1e-9

NAVY, BLUE, GREEN, GREY, LIGHT, ORANGE = ("#1f4e79", "#5b9bd5", "#70ad47", "#9a9a9a",
                                          "#d0d0d0", "#ed7d31")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
})


def rules_benchmark():
    df = pd.read_csv(os.path.join(RES, "benchmark_overall.csv")).sort_values("overall")
    colors = [NAVY if r.startswith("WSPT") else ORANGE if r == "Random"
              else BLUE if r.startswith("M") and r != "MaxWeight" and r != "MinSlack"
              else GREY for r in df.rule]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    y = np.arange(len(df))[::-1]
    ax.barh(y, df.overall, color=colors, height=0.7)
    for yi, v in zip(y, df.overall):
        ax.text(v + 0.5, yi, f"{v:.2f}", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df.rule)
    ax.set_xlabel("mean gap to the LP bound (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, df.overall.max() * 1.15)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "rules_benchmark.pdf"))
    plt.close(fig)


# worst gap to the exact optimum found per family, report 05's table
ADV_WORST = [("bimodal p", 60.9, "ranking, decoder"), ("w correlated", 42.1, "decoder, ranking"),
             ("narrow R=0.1", 33.3, "decoder, ranking"), ("lognormal p", 27.8, "decoder"),
             ("loose T=0.1", 27.8, "decoder"), ("tight T=0.8", 26.5, "ranking"),
             ("in-dist", 26.3, "decoder"), ("single m=1", 25.7, "ranking"),
             ("many machines", 22.9, "decoder"), ("large n", 22.5, "ranking"),
             ("w anticorr", 17.9, "decoder"), ("scale-shift", 5.6, "decoder, ranking")]
# plain names for the families, as the report text describes them
ADV_LABEL = {"bimodal p": "bimodal job lengths", "w correlated": "weights grow with length",
             "narrow R=0.1": "narrow deadline window", "lognormal p": "lognormal job lengths",
             "loose T=0.1": "loose deadlines", "tight T=0.8": "very tight deadlines",
             "in-dist": "base setting", "single m=1": "single machine",
             "many machines": "many machines (16)", "large n": "very many jobs (500)",
             "w anticorr": "weights shrink with length",
             "scale-shift": "many jobs (200 to 300)"}

# mean % gap, four machines, exactly as printed in report 09's table
SIM_CASES = ["none similar", "p similar", "d similar", "w similar", "p & d similar",
             "p & w similar", "d & w similar", "p & d & w similar"]
SIM_M4 = {
    "WSPT-EDD":       [2.64, 1.70, 4.42, 3.63, 0.81, 3.69, 4.20, 2.08],
    "WSPT-Laxity":    [3.01, 1.80, 4.55, 3.52, 0.83, 3.75, 4.23, 2.20],
    "MooreHodgson-W": [3.30, 1.76, 5.75, 3.90, 0.81, 3.67, 4.73, 2.25],
    "Density-wpd":    [3.61, 2.66, 4.42, 4.28, 0.88, 5.34, 4.00, 2.58],
    "MaxWeight":      [3.63, 1.71, 7.83, 11.02, 0.69, 4.57, 20.86, 0.55],
}


def adversarial_worst():
    fam = ADV_WORST
    col = {"ranking": BLUE, "decoder": NAVY}
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = np.arange(len(fam))[::-1]
    for yi, (name, gap, cause) in zip(y, fam):
        ax.barh(yi, gap, height=0.7, color=col[cause.split(",")[0]])
        ax.text(gap + 0.8, yi, f"{gap:.1f}", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([ADV_LABEL[f[0]] for f in fam])
    ax.set_xlabel("worst gap to the optimum found (%)")
    ax.set_xlim(0, 70)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=NAVY, label="mainly decoder failure"),
                       Patch(color=BLUE, label="mainly ranking failure")],
              loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "adversarial_worst.pdf"))
    plt.close(fig)


def decoder_failure(name="loose_T0.1__WSPT"):
    J = json.load(open(os.path.join(RES, "metaopt", name + ".json")))
    inst = P1Instance(p=J["p"], d=J["d"], w=J["w"], m=J["m"])
    keep = accepted_mask(inst, RULES["WSPT"].fn, RULES["WSPT"].decoder)
    rows = sorted((inst.d[i], inst.p[i], i) for i in np.where(keep)[0])
    wspt = [(i, j, s, e) for i, j, s, e in schedule_of(rows, inst.m)]
    opt = []
    for j in range(inst.m):
        t = 0.0
        for i in sorted((int(k) for k, v in J["opt_assign"].items() if v == j),
                        key=lambda i: inst.d[i]):
            opt.append((i, j, t, t + inst.p[i]))
            t += inst.p[i]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.2), sharex=True)
    for ax, sched, title in [(axes[0], wspt, f"WSPT: J3 rejected, loss {J['rules']['WSPT']['lost']:.0f}"),
                             (axes[1], opt, f"Optimum: all jobs on time, loss {J['opt_lost']:.0f}")]:
        for i, j, s, e in sched:
            late_only = i not in [x[0] for x in wspt]
            ax.barh(j, e - s, left=s, height=0.6, color=BLUE if late_only else LIGHT,
                    edgecolor="white")
            ax.text((s + e) / 2, j, f"J{i + 1}", ha="center", va="center", fontsize=7)
        ax.axvline(inst.d[2], color=ORANGE, ls="--", lw=1)
        ax.set_yticks(range(inst.m))
        ax.set_yticklabels([f"M{j + 1}" for j in range(inst.m)])
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("time (dashed line: deadline of J3)")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "decoder_failure.pdf"))
    plt.close(fig)


def similarity_m4():
    rules = list(SIM_M4)
    cols = [NAVY, BLUE, GREEN, GREY, ORANGE]
    cases = SIM_CASES
    g = pd.DataFrame(SIM_M4, index=cases).T
    cap, width, x = 7.0, 0.16, np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(7.16, 1.9))
    for k, (r, col) in enumerate(zip(rules, cols)):
        v = g.loc[r].to_numpy()
        xs = x + (k - 2) * width
        ax.bar(xs, np.minimum(v, cap), width, color=col, label=r)
        for xi, vi in zip(xs, v):
            if vi > cap:
                ax.text(xi, cap + 0.1, f"{vi:.0f}", ha="center", va="bottom", fontsize=6,
                        color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" similar", "").replace(" & ", ", ") for c in cases])
    ax.set_xlabel("quantities that are similar")
    ax.set_ylabel("mean gap to optimum (%)")
    ax.set_ylim(0, cap + 0.9)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "similarity_m4.pdf"))
    plt.close(fig)


def ga_ablation():
    a = pd.read_csv(os.path.join(RES, "ga", "adversarial.csv"))
    a = a[a["WSPT"] > a["opt"] + EPS]
    meths = [("WSPT", GREY), ("best_rule", LIGHT), ("GA-priority", BLUE),
             ("GA-assignment", GREEN), ("GA-joint", NAVY)]
    labels = {"best_rule": "best of 17 rules"}
    groups = [("ranking", "ranking failures"), ("decoder", "decoder failures")]
    width, x = 0.16, np.arange(2)
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    for k, (m, col) in enumerate(meths):
        v = [a[a.cause == c][m + " gap%"].mean() for c, _ in groups]
        xs = x + (k - 2) * width
        ax.bar(xs, v, width, color=col, label=labels.get(m, m))
        for xi, vi in zip(xs, v):
            if vi < 0.05:
                ax.text(xi, 0.4, "0", ha="center", fontsize=6, color=col)
    n = [int((a.cause == c).sum()) for c, _ in groups]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} ({k})" for (_, lab), k in zip(groups, n)])
    ax.set_ylabel("mean gap to optimum (%)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.32))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "ga_ablation.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    rules_benchmark()
    adversarial_worst()
    decoder_failure()
    similarity_m4()
    ga_ablation()
    print(f"figures -> {os.path.abspath(OUT)}")
