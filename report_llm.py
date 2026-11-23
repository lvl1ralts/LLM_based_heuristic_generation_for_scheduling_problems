#!/usr/bin/env python3
"""
Figure for the LLM-heuristics demo report (Findings/13_LLM_Heuristics_Demo).

  python report_llm.py

Reads results/llm/<model>/test.csv and writes results/llm/figures/test_gaps.png:
mean gap per test set for WSPT, the two hand-written references on the new
decoder, the best heuristic of each model, and GA-joint (report 10).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "llm")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

MODELS = [("gemini-3.7-flash", "Gemini 3.7 Flash"), ("gemini-3.8-flash", "Gemini 3.8 Flash"),
          ("gemma-4-31b-it", "Gemma 4 31B")]
GROUPS = [("adversarial", "Adversarial (24)"), ("similarity m=1", "Similarity, 1 machine (40)"),
          ("similarity m=4", "Similarity, 4 machines (40)"), ("benchmark", "Benchmark (24)")]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["IBM Plex Sans", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e6e6", "axes.axisbelow": True, "legend.frameon": False,
})


def main():
    T = {m: pd.read_csv(os.path.join(RES, m, "test.csv")) for m, _ in MODELS}
    base = T[MODELS[0][0]]
    bars = [("WSPT", "WSPT", "#9a9a9a"),
            ("seed", "WSPT, first machine that fits (seed)", "#c9c9c9"),
            ("WSPT_bestfit", "WSPT, tightest fit (hand-written)", "#e3e3e3")]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.7))
    for ax, (g, title) in zip(axes, GROUPS):
        b = base[base.group == g]
        vals = [(lab, b[k + " gap%"].mean(), col) for k, lab, col in bars]
        for (m, lab), col in zip(MODELS, ["#3446b8", "#7d8de0", "#1f6b33"]):
            s = T[m][T[m].group == g]
            vals.append((lab, s["LLM gap%"].mean(), col))
        vals.append(("GA-joint (report 10)", b["GA_joint gap%"].mean(), "#2b2f3a"))
        x = range(len(vals))
        ax.bar(x, [v for _, v, _ in vals], color=[c for _, _, c in vals], width=0.78,
               edgecolor=["#bbbbbb" if c == "#e3e3e3" else c for _, _, c in vals], lw=0.6)
        top = max(v for _, v, _ in vals)
        for xi, (_, v, _) in zip(x, vals):
            ax.text(xi, v + top * 0.02, f"{v:.2f}" if v < 10 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6.3, rotation=90)
        ax.set_ylim(0, top * 1.28)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=8.3)
    axes[0].set_ylabel("mean gap to reference (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="#bbbbbb" if c == "#e3e3e3" else c)
               for _, _, c in vals]
    fig.legend(handles, [lab for lab, _, _ in vals], ncol=4, loc="lower center",
               fontsize=7.4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.8)
    fig.savefig(os.path.join(OUT, "test_gaps.png"), dpi=220)
    plt.close(fig)
    print(f"figure -> {OUT}")


CMP = [("WSPT", "WSPT", "#9a9a9a"), ("best_of_17", "best of 17 rules, per instance", "#c9c9c9"),
       ("seed", "WSPT seed, new decoder", "#e3e3e3"),
       ("Gemini 3.8 Flash", "Gemini 3.8 Flash", "#3446b8"), ("Gemma 4 31B", "Gemma 4 31B", "#1f6b33"),
       ("GPT-OSS 120B", "GPT-OSS 120B (Groq)", "#c25a1c"), ("Qwen 3.8 27B", "Qwen 3.8 27B (Groq)", "#8a3fa0"),
       ("GA_joint", "GA-joint (report 10)", "#2b2f3a")]


def compare():
    """Report 14 figures, from results/llm/outputs (llm_compare.py)."""
    S = pd.read_csv(os.path.join(RES, "outputs", "summary_by_set.csv"))
    a = S[S.group == "all"].set_index("set")
    panels = [("family", "Families, 102 (LP bound)"), ("adversarial", "Adversarial, 24"),
              ("similarity m=1", "Similarity, 1 machine, 40"),
              ("similarity m=4", "Similarity, 4 machines, 40")]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.8))
    for ax, (s, title) in zip(axes, panels):
        vals = [(lab, a.loc[s, k], col) for k, lab, col in CMP
                if not (s == "family" and k == "GA_joint")]
        x = range(len(vals))
        ax.bar(x, [v for _, v, _ in vals], color=[c for _, _, c in vals], width=0.78,
               edgecolor=["#bbbbbb" if c == "#e3e3e3" else c for _, _, c in vals], lw=0.6)
        top = max(v for _, v, _ in vals)
        for xi, (_, v, _) in zip(x, vals):
            ax.text(xi, v + top * 0.02, f"{v:.2f}" if v < 10 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6.3, rotation=90)
        ax.set_ylim(0, top * 1.3)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=8.3)
    axes[0].set_ylabel("mean gap (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="#bbbbbb" if c == "#e3e3e3" else c)
               for _, _, c in CMP]
    fig.legend(handles, [lab for _, lab, _ in CMP], ncol=4, loc="lower center",
               fontsize=7.4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.8)
    fig.savefig(os.path.join(OUT, "compare_sets.png"), dpi=220)
    plt.close(fig)

    F = S[(S.set == "family") & (S.group != "all")].set_index("group")
    fams = list(F.index)
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    keys = [c for c in CMP if c[0] not in ("seed", "GA_joint")]
    wdt = 0.8 / len(keys)
    for k, (key, lab, col) in enumerate(keys):
        ax.bar([i + (k - (len(keys) - 1) / 2) * wdt for i in range(len(fams))], F[key], width=wdt,
               color=col, label=lab, edgecolor=col, lw=0.4)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f.replace("shift: ", "").replace(" n50-100", "").replace(" n200-300", "")
                        .replace("weights ", "w ").replace("narrow due range", "narrow")
                        for f in fams], rotation=30, ha="right", fontsize=7.4)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("mean gap to LP bound (%)")
    ax.text(fams.index("loose T=0.1"), 0.15, "all 0", ha="center", fontsize=7, color="#8a8f9c")
    ax.legend(ncol=6, fontsize=7.0, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "compare_families.png"), dpi=220)
    plt.close(fig)
    print(f"report 14 figures -> {OUT}")


if __name__ == "__main__":
    import sys
    compare() if "--compare" in sys.argv else main()
