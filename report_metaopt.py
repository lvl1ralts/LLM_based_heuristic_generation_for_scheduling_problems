#!/usr/bin/env python3
"""
Explain and plot the adversarial instances found by metaopt.py.

For every instance:
  * which jobs the rule kept that the optimum dropped ("displacers") and the
    reverse ("victims")
  * the cause: 'ranking' if the optimal job set passes the decoder's own
    feasibility test (so a better priority order could reach it), 'decoder'
    if it does not (no priority rule can reach it)
  * a plain-language explanation of the mechanism
  * a figure: the rule's schedule next to the optimal schedule

Writes results/metaopt/summary.csv, results/metaopt/explanations.md and
results/metaopt/figures/*.png.
"""
from __future__ import annotations

import glob
import heapq
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduling.heuristics import RULES

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "metaopt")
FIG = os.path.join(SRC, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#b9bec9", "#1a1d29", "#6b7180"
RULE_ORDER = ["WSPT", "WSPT-EDD", "WSPT-Laxity"]
BENCH_NAME = {"in-dist": "in-dist n50-100", "scale-shift": "scale-shift n200-300",
              "bimodal p": "shift: bimodal p", "lognormal p": "shift: lognormal p",
              "tight T=0.8": "shift: very tight T=0.8", "loose T=0.1": "loose T=0.1",
              "w correlated": "weights correlated", "w anticorr": "weights anticorrelated",
              "narrow R=0.1": "narrow due range R=0.1", "many machines": "many machines m=16",
              "single m=1": "single machine m=1", "large n": "large scale n=500"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c8d3",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def priority(rule, p, d, w):
    if rule == "WSPT-Laxity":
        return w / p * (1 + (d - p) / d)
    return w / p


def list_schedule(idx, p, d, m):
    """EDD list scheduling onto the earliest-free machine (the decoder's test)."""
    heap = [(0.0, j) for j in range(m)]
    out = []
    for i in sorted(idx, key=lambda i: (d[i], p[i])):
        t, j = heapq.heappop(heap)
        out.append((i, j, t, t + p[i]))
        heapq.heappush(heap, (t + p[i], j))
    return out


def opt_schedule(assign, p, d):
    out = []
    for j in sorted(set(assign.values())):
        t = 0.0
        for i in sorted([i for i, mj in assign.items() if mj == j], key=lambda i: d[i]):
            out.append((i, j, t, t + p[i]))
            t += p[i]
    return out


def describe(r, rule):
    p, d, w = np.array(r["p"]), np.array(r["d"]), np.array(r["w"])
    m = r["m"]
    keep = np.array(r["rules"][rule]["keep"])
    opt = np.array(r["opt_keep"])
    disp = [int(i) for i in np.where(keep & ~opt)[0]]
    vict = [int(i) for i in np.where(opt & ~keep)[0]]
    pr = priority(rule, p, d, w)
    job = lambda i: f"job {i} (p={p[i]:.0f}, d={d[i]:.0f}, w={w[i]:.0f}, priority {pr[i]:.3f})"
    lines, tags = [], set()
    if r["cause"] == "ranking":
        tags.add("ranking")
        for v in vict:
            blockers = [x for x in disp if pr[x] > pr[v] and d[x] <= d[v] + 1e-9] or \
                       [x for x in disp if pr[x] > pr[v]]
            # replay the decoder's actual decision on v: jobs admitted before it, plus v
            before_v = [int(x) for x in np.where(keep)[0] if pr[x] > pr[v]]
            miss = [(i, e) for i, _, _, e in list_schedule(before_v + [v], p, d, m)
                    if e > d[i] + 1e-9]
            if blockers:
                b = blockers[0]
                if w[b] < w[v]:
                    tags.add("low-value job ranked ahead of high-value job")
                if d[b] <= d[v]:
                    tags.add("blocker holds capacity before the victim's deadline")
                why = (f"admitting {v} on top of the jobs already accepted would make job "
                       f"{miss[0][0]} finish at {miss[0][1]:.0f}, after its deadline "
                       f"{d[miss[0][0]]:.0f}" if miss else "it no longer fits")
                lines.append(f"{rule} ranks {job(b)} ahead of {job(v)}. When {v}'s turn comes, "
                             f"{why}, so {v} is rejected. The optimum makes the opposite choice: "
                             f"it drops {b} and keeps {v}, worth {w[v]:.0f} instead of {w[b]:.0f}.")
            else:
                lines.append(f"{job(v)} is rejected; the optimum keeps it.")
        if len(disp) > 1 or len(vict) > 1:
            tags.add("several jobs traded")
    else:
        tags.add("decoder")
        sched = list_schedule([int(i) for i in np.where(opt)[0]], p, d, m)
        misses = [(i, j, s, e) for i, j, s, e in sched if e > d[i] + 1e-9]
        if misses:
            i, j, s, e = misses[0]
            osch = {x: (mj, a, b) for x, mj, a, b in
                    opt_schedule({int(k): v for k, v in r["opt_assign"].items()}, p, d)}
            om, oa, ob = osch[i]
            mates = [x for x, (mj, _, _) in osch.items() if mj == om and x != i]
            lines.append(f"The optimum keeps this job set by giving {job(i)} its own room: it runs "
                         f"on machine {om + 1} from {oa:.0f} to {ob:.0f}"
                         + (f", sharing it only with job(s) {mates}" if mates else ", alone") + ". "
                         f"The decoder's test instead places every job, in deadline order, on the "
                         f"machine that becomes free first. That spreads the earlier jobs over all "
                         f"machines, so the earliest any machine is free for job {i} is {s:.0f}, "
                         f"and it would finish at {e:.0f}, after its deadline {d[i]:.0f}. The test "
                         f"rejects the set, so no priority order can make the decoder keep it.")
            tags.add("earliest-free placement fragments machine capacity")
        bigp = [i for i in np.where(opt)[0] if p[i] >= 0.5 * p.max()]
        if bigp:
            tags.add("long job must sit alone on a machine")
        if len(vict):
            lines.append("Jobs lost as a result: " + "; ".join(job(v) for v in vict) +
                         (". Kept instead: " + "; ".join(job(x) for x in disp) if disp else "") + ".")
    return disp, vict, lines, sorted(tags)


def gantt(ax, sched, m, d, w, colour_of, title):
    for i, j, s, e in sched:
        ax.barh(j, e - s, left=s, height=0.62, color=colour_of(i), edgecolor="white", linewidth=1.5)
        ax.text((s + e) / 2, j, f"{i}", ha="center", va="center", fontsize=8,
                color="white" if colour_of(i) != GREY else INK)
        ax.plot([d[i], d[i]], [j - 0.36, j + 0.36], color=INK, lw=0.8, alpha=0.55)
    ax.set_yticks(range(m))
    ax.set_yticklabels([f"M{j+1}" for j in range(m)])
    ax.set_ylim(-0.6, m - 0.4)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(axis="x", color="#e4e7ee", lw=0.6)
    ax.set_axisbelow(True)


def figure(r, rule, disp, vict, path):
    p, d, w = np.array(r["p"]), np.array(r["d"]), np.array(r["w"])
    m = r["m"]
    keep = np.array(r["rules"][rule]["keep"])
    opt_assign = {int(k): v for k, v in r["opt_assign"].items()}

    def colour(i):
        return ORANGE if i in disp else BLUE if i in vict else GREY

    rs = list_schedule([int(i) for i in np.where(keep)[0]], p, d, m)
    os_ = opt_schedule(opt_assign, p, d)
    xmax = max([e for *_, e in rs + os_] + list(d)) * 1.05
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 1.0 + 0.55 * m * 2 + 0.6), sharex=True)
    lost_r, lost_o = r["rules"][rule]["lost"], r["opt_lost"]
    gantt(axes[0], rs, m, d, w, colour, f"{rule}: keeps {sorted(np.where(keep)[0].tolist())}, "
                                        f"loses weight {lost_r:.0f}")
    gantt(axes[1], os_, m, d, w, colour, f"Optimum: keeps {sorted(opt_assign)}, "
                                         f"loses weight {lost_o:.0f}")
    axes[1].set_xlim(0, xmax)
    axes[1].set_xlabel("time  (ticks mark each job's deadline)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, ORANGE, BLUE)]
    fig.legend(handles, ["kept by both", f"kept only by {rule}", "kept only by optimum"],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{r['family']}  |  n={r['n']}, m={m}  |  gap {r['rules'][rule]['gap_pct']:.1f}%",
                 x=0.01, ha="left", fontsize=10, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(SRC, "*__*.json")))]
    rows, md = [], ["# Adversarial instances: explanations\n"]
    for r in recs:
        rule = r["encoded_rule"]
        disp, vict, lines, tags = describe(r, rule)
        r["tags"] = tags
        stem = f"{r['family'].replace(' ', '_').replace('=', '')}__{rule}"
        figure(r, rule, disp, vict, os.path.join(FIG, stem + ".png"))
        row = dict(family=r["family"], built_against=rule, n=r["n"], m=r["m"],
                   total_w=r["total_w"], opt_lost=r["opt_lost"], cause=r["cause"],
                   search_proved_worst=r["solver_proved"], encoding_verified=r["encoding_verified"],
                   tags="; ".join(tags))
        for rn in RULE_ORDER:
            row[f"{rn}_lost"] = r["rules"][rn]["lost"]
            row[f"{rn}_gap"] = round(r["rules"][rn]["gap_pct"], 2)
        rows.append(row)
        jobs = "\n".join(f"| {i} | {r['p'][i]:.0f} | {r['d'][i]:.0f} | {r['w'][i]:.0f} | "
                         f"{'yes' if r['rules'][rule]['keep'][i] else 'no'} | "
                         f"{'yes' if r['opt_keep'][i] else 'no'} |" for i in range(r["n"]))
        md += [f"\n## {r['family']}, built against {rule}\n",
               f"n={r['n']}, m={r['m']}, total weight {r['total_w']:.0f}. "
               f"{rule} loses {r['rules'][rule]['lost']:.0f}, the optimum loses {r['opt_lost']:.0f} "
               f"(gap {r['rules'][rule]['gap_pct']:.2f}%). Cause: **{r['cause']}**. "
               f"Pattern: {', '.join(tags)}.\n",
               "| job | p | d | w | rule keeps | optimum keeps |\n|---|---|---|---|---|---|\n" + jobs + "\n",
               *[l + "\n" for l in lines]]
    df = pd.DataFrame(rows).sort_values(["family", "built_against"])
    df.to_csv(os.path.join(SRC, "summary.csv"), index=False)
    open(os.path.join(SRC, "explanations.md"), "w", encoding="utf-8").write("\n".join(md))

    # overview figure: worst gap found per family and rule, against the benchmark mean
    bench = pd.read_csv(os.path.join(HERE, "results", "benchmark_by_family.csv"))
    bench = bench[bench.rule == "WSPT"].iloc[0]
    fams = [f for f in BENCH_NAME if f in set(df.family)]
    worst = {rn: [df[df.family == f][f"{rn}_gap"].max() for f in fams] for rn in RULE_ORDER}
    yy = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(fams) + 1.2))
    for k, (rn, col) in enumerate(zip(RULE_ORDER, (BLUE, "#7aa7e6", ORANGE))):
        ax.barh(yy + (k - 1) * 0.26, worst[rn], height=0.24, color=col, label=rn)
    ax.scatter([bench[BENCH_NAME[f]] for f in fams], yy, marker="|", s=260, color=INK, zorder=3,
               label="benchmark mean (WSPT, random instances)")
    ax.set_yticks(yy)
    ax.set_yticklabels(fams)
    ax.invert_yaxis()
    ax.set_xlabel("gap to optimum (%)  on the worst instance found")
    ax.grid(axis="x", color="#e4e7ee", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "overview.png"), dpi=170)
    plt.close(fig)
    print(df.drop(columns=["tags"]).to_string(index=False))
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
