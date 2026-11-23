#!/usr/bin/env python3
"""
Figure for report 09, from the CSV that similarity_cases.py writes.

  python report_similarity.py

Writes results/similarity/figures/leading_rules.png: the five leading rules,
mean % gap per similarity case on four machines (bars above 7% are cut and
labelled).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "similarity")
FIG = os.path.join(SRC, "figures")
os.makedirs(FIG, exist_ok=True)

RULES = ["WSPT-EDD", "WSPT-Laxity", "MooreHodgson-W", "Density-wpd", "MaxWeight"]
COLOR = ["#1f4e79", "#5b9bd5", "#70ad47", "#a5a5a5", "#ed7d31"]
CASES = ["none similar", "p similar", "d similar", "w similar", "p & d similar",
         "p & w similar", "d & w similar", "p & d & w similar"]
CAP = 7.0

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["IBM Plex Sans", "Segoe UI", "Arial"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e6e6",
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 200,
})


def leading_rules():
    df = pd.read_csv(os.path.join(SRC, "detail.csv"))
    df = df[df.setting == "m=4, n=40"]
    g = df.pivot_table(index="rule", columns="case", values="mean_gap")[CASES]
    x = np.arange(len(CASES))
    width = 0.8 / len(RULES)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, (r, col) in enumerate(zip(RULES, COLOR)):
        v = g.loc[r].to_numpy()
        xs = x + (i - (len(RULES) - 1) / 2) * width
        ax.bar(xs, np.minimum(v, CAP), width, color=col, label=r)
        for xi, vi in zip(xs, v):
            if vi > CAP:
                ax.text(xi, CAP + 0.1, f"{vi:.0f}", ha="center", va="bottom", fontsize=7,
                        color="#c55a11")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" similar", "").replace(" & ", ",") for c in CASES])
    ax.set_xlabel("similar numbers")
    ax.set_ylabel("mean % gap")
    ax.set_ylim(0, CAP + 0.8)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "leading_rules.png"))
    plt.close(fig)


if __name__ == "__main__":
    leading_rules()
    print(f"figure -> {FIG}")
