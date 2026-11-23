#!/usr/bin/env python3
"""
Report 14: the best heuristic each LLM wrote, on every instance set we have.

  python llm_compare.py

No API calls. Reads the runs in results/llm/<model>/candidates.jsonl and, for
each model, takes the heuristic with the lowest training gap that the model
wrote itself (the hand-written seed is left out, so a model that never beat the
seed is still shown with its own best attempt). Each one is then run on

  family       all 102 instances of the 12 benchmark families (reports 01-04),
               gap to the LP lower bound, as in those reports
  adversarial  the 24 MetaOpt instances (reports 05-08), gap to the exact optimum
  similarity   80 instances, 5 per case x 8 cases, m=1 and m=4 (report 09),
               gap to the exact optimum

next to all 17 priority rules (each with its own decoder, as in reports 01-04),
the WSPT seed on the new decoder, and GA-joint (report 10) where it was run.

Writes results/llm/outputs/:
  README.md                 what is in the folder, one line per model
  per_instance.csv          every instance x every method (weight lost, gap)
  summary_by_set.csv        mean gap per set (and per family / similarity case)
  <model>/heuristic.py      the heuristic that was tested, with its idea
  <model>/all_heuristics.md every heuristic the model wrote, with its training gap
  <model>/per_instance.csv  this model next to WSPT, the best rule and GA-joint
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm_heuristics as L
from scheduling.environment import evaluate, lower_bound
from scheduling.heuristics import RULES
from scheduling.instances import P1Instance
from scheduling.suites import all_suites

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "llm")
OUT = os.path.join(RES, "outputs")

# (results folder, label, how it was called)
MODELS = [
    ("gemini-3.8-flash", "Gemini 3.8 Flash", "Google Gemini API (Gemini key)"),
    ("gemma-4-31b-it", "Gemma 4 31B", "Google Gemini API (Gemini key); open-weights model"),
    ("groq_openai_gpt-oss-120b", "GPT-OSS 120B", "Groq API (Groq key); OpenAI open-weights model"),
    ("groq_qwen_qwen3.8-27b", "Qwen 3.8 27B", "Groq API (Groq key); Alibaba open-weights model"),
]
FOLDER = {m: lab.replace(" ", "_") for m, lab, _ in MODELS}


def api_error(h):
    """The call itself failed (network, HTTP, quota): the model wrote nothing."""
    e = h["error"] or ""
    return e.startswith(("RuntimeError", "RemoteDisconnected", "URLError", "TimeoutError"))


def best_own(model):
    log = [json.loads(l) for l in open(os.path.join(RES, model, "candidates.jsonl"), encoding="utf-8")]
    own = [h for h in log if h["op"] != "seed" and not api_error(h)]
    ok = [h for h in own if h["fitness"] is not None]
    seed = next(h for h in log if h["op"] == "seed")
    return min(ok, key=lambda h: h["fitness"]), log, len(own), len(ok), seed["fitness"]


# ------------------------------------------------------------------------- #
# running a heuristic's code in a separate process (it is LLM-written)
# ------------------------------------------------------------------------- #
def _run(code, data):
    prio_fn, choose = L.compile_code(code)
    out = []
    for p, d, w, m in data:
        inst = P1Instance(p=p, d=d, w=w, m=m)
        f, c = L.Fields(L.features(inst)), L.Fields(L.context(inst))
        prio = np.nan_to_num(np.asarray(prio_fn(f, c), dtype=float).ravel(),
                             nan=0.0, posinf=1e12, neginf=-1e12)
        lost, mach = L.decode(inst, prio, choose)
        out.append((lost, int((mach < 0).sum())))
    return out


def _worker(code, data, q):
    try:
        q.put(("ok", _run(code, data)))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"[:400]))


def run_isolated(code, data, timeout=1800):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(code, data, q), daemon=True)
    proc.start()
    status, val = q.get(timeout=timeout)
    proc.join(1)
    if proc.is_alive():
        proc.kill()
    if status != "ok":
        raise RuntimeError(val)
    return val


# ------------------------------------------------------------------------- #
# instances
# ------------------------------------------------------------------------- #
def instances():
    """[(set, subgroup, instance name, inst, ref, ref_kind, GA-joint lost or nan)]"""
    G = os.path.join(HERE, "results", "ga")
    rows = []
    ga_b = pd.read_csv(os.path.join(G, "benchmark.csv")).set_index("instance")["GA-joint"]
    for fam, insts in all_suites().items():
        for i in insts:
            rows.append(("family", fam, i.name, i, lower_bound(i), "LP bound",
                         float(ga_b.get(i.name, np.nan))))
    sets = L.test_sets()
    adv = pd.read_csv(os.path.join(G, "adversarial.csv")).set_index("instance")
    insts, base = sets["adversarial"]
    for i, b in zip(insts, base):
        rows.append(("adversarial", adv.loc[b["instance"], "family"], b["instance"], i,
                     b["ref"], "optimum", b["GA_joint"]))
    for lab in ["similarity m=1", "similarity m=4"]:
        insts, base = sets[lab]
        for i, b in zip(insts, base):
            case, name = b["instance"].split("|")
            rows.append((lab, case, name, i, b["ref"], "optimum", b["GA_joint"]))
    return rows


def gap(lost, ref, W):
    return 100.0 * (lost - ref) / max(W - ref, L.EPS)


# ------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    R = instances()
    data = [(i.p, i.d, i.w, i.m) for _, _, _, i, _, _, _ in R]
    print(f"{len(R)} instances: " + ", ".join(
        f"{s} {sum(r[0] == s for r in R)}" for s in dict.fromkeys(r[0] for r in R)), flush=True)

    df = pd.DataFrame([dict(set=s, group=g, instance=nm, n=i.n, m=i.m, W=i.total_weight,
                            ref=ref, ref_kind=k) for s, g, nm, i, ref, k, _ in R])

    # all 17 rules, each with its own decoder (as in reports 01-04)
    for r in RULES:
        df[r] = [evaluate(i, RULES[r].fn, RULES[r].decoder) for _, _, _, i, _, _, _ in R]
    print(f"  rules done ({time.time() - t0:.0f}s)", flush=True)
    rule_cols = list(RULES)
    df["best_of_17"] = df[rule_cols].min(axis=1)
    df["best_of_17_name"] = df[rule_cols].idxmin(axis=1)
    df["GA_joint"] = [r[6] for r in R]

    # the WSPT seed on the new decoder, then each model's own best heuristic
    seed_code = L.parse(L.SEED)[1]
    res = run_isolated(seed_code, data)
    df["seed"] = [x[0] for x in res]
    info = []
    for model, label, how in MODELS:
        best, log, n_written, n_ok, seed_fit = best_own(model)
        res = run_isolated(best["code"], data)
        df[label] = [x[0] for x in res]
        df[label + " late jobs"] = [x[1] for x in res]
        info.append(dict(model=model, label=label, how=how, best=best, log=log,
                         written=n_written, working=n_ok, seed_fit=seed_fit))
        print(f"  {label} done ({time.time() - t0:.0f}s)", flush=True)

    methods = rule_cols + ["best_of_17", "seed"] + [m["label"] for m in info] + ["GA_joint"]
    for k in methods:
        df[k + " gap%"] = [gap(l, r, w) if np.isfinite(l) else np.nan
                           for l, r, w in zip(df[k], df.ref, df.W)]
    df.round(4).to_csv(os.path.join(OUT, "per_instance.csv"), index=False)

    # mean gap per set and per subgroup. The family set is averaged over its 12
    # families (as report 01 does); "family, GA's 24" is the subset GA-joint ran on.
    ga24 = df[(df.set == "family") & df.GA_joint.notna()].assign(set="family, GA's 24")
    rows = []
    parts = [((s, "all"), sub) for s, sub in df.groupby("set", sort=False)] + \
        [(("family, GA's 24", "all"), ga24)] + list(df.groupby(["set", "group"], sort=False))
    for (s, g), sub in parts:
        r = dict(set=s, group=g, instances=len(sub))
        for k in methods:
            if s == "family" and g == "all":
                v = sub.groupby("group")[k + " gap%"].mean()
            else:
                v = sub[k + " gap%"].dropna()
            r[k] = round(v.mean(), 3) if len(v) else np.nan
        exact = not s.startswith("family")
        for k in ["WSPT", "best_of_17", "seed"] + [m["label"] for m in info] + ["GA_joint"]:
            r[k + " optimal on"] = int((sub[k] <= sub.ref + 1e-6).sum()) if exact else np.nan
        r["GA_joint instances"] = int(sub["GA_joint"].notna().sum())
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "summary_by_set.csv"), index=False)

    # which half does the work? Swap each half with the seed's (a later def wins)
    first_fit = "\n\ndef choose_machine(i, f, c, M):\n    return -np.arange(len(M[\"fits\"]))\n"
    wspt = "\n\ndef priority(f, c):\n    return f[\"wp\"]\n"
    ab = df[["set", "group", "ref", "W"]].copy()
    variants = {"seed (WSPT + first fit)": seed_code}
    for m in info:
        variants[f"{m['label']}: own priority + first fit"] = m["best"]["code"] + first_fit
        variants[f"{m['label']}: WSPT + own placement"] = m["best"]["code"] + wspt
        variants[f"{m['label']}: both (as written)"] = m["best"]["code"]
    arows = []
    for name, code in variants.items():
        lost = [x[0] for x in run_isolated(code, data)]
        ab["g"] = [gap(l, r, w) for l, r, w in zip(lost, ab.ref, ab.W)]
        r = dict(variant=name)
        for s, sub in ab.groupby("set", sort=False):
            r[s] = round((sub.groupby("group").g.mean().mean() if s == "family" else sub.g.mean()), 3)
        arows.append(r)
    pd.DataFrame(arows).to_csv(os.path.join(OUT, "swap_halves.csv"), index=False)
    print(pd.DataFrame(arows).to_string(index=False))

    # per-model folders
    for m in info:
        write_model(m, df)
    write_readme(info, S)
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


def write_model(m, df):
    d = os.path.join(OUT, FOLDER[m["model"]])
    os.makedirs(d, exist_ok=True)
    b, lab = m["best"], m["label"]
    with open(os.path.join(d, "heuristic.py"), "w", encoding="utf-8") as fh:
        fh.write(f'"""{lab}: the heuristic tested in report 14.\n\n'
                 f"Called through: {m['how']}\n"
                 f"Written in generation {b['gen']} by operator {b['op']} "
                 f"(heuristic #{b['id']} of the run).\n"
                 f"Training gap {b['fitness']:.3f}% (WSPT seed: {m['seed_fit']:.3f}%, "
                 f"WSPT: 4.183%).\n\n"
                 f"The model's own idea:\n  {b['idea']}\n\n"
                 "priority(f, c): one score per job, the jobs are tried highest first.\n"
                 "choose_machine(i, f, c, M): one score per machine for job i; the job goes on\n"
                 "the machine that fits with the highest score, or is late if none fits.\n"
                 '"""\nimport math\n\nimport numpy as np\n\n\n' + b["code"] + "\n")
    ops = {"seed": "hand-written start (WSPT, first fit)", "i1": "new from scratch",
           "e1": "new idea, unlike its parents", "e2": "new idea from what its parents share",
           "m1": "changed a parent's logic", "m2": "changed a parent's constants"}
    lines = [f"# Every heuristic {lab} wrote\n",
             f"Called through {m['how']}. {m['written']} heuristics written, {m['working']} ran "
             f"correctly. Training gap = mean % gap to the LP bound on the 12 training "
             f"instances (lower is better). WSPT scores 4.183%, the hand-written seed "
             f"{m['seed_fit']:.3f}%.\n",
             f"The one marked **tested** has the lowest training gap among the model's own "
             f"heuristics and is the one run in report 14.\n"]
    for h in m["log"]:
        tag = " — **tested**" if h["id"] == b["id"] else ""
        fit = f"training gap {h['fitness']:.3f}%" if h["fitness"] is not None \
            else f"API call failed, nothing written: {h['error'][:80]}" if api_error(h) \
            else f"rejected: {h['error']}"
        lines.append(f"\n## #{h['id']} · generation {h['gen']} · {h['op']} "
                     f"({ops.get(h['op'], h['op'])}) · {fit}{tag}\n")
        if h["idea"]:
            lines.append(f"\n*Idea:* {h['idea']}\n")
        if h["code"]:
            lines.append(f"\n```python\n{h['code']}\n```\n")
    with open(os.path.join(d, "all_heuristics.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    cols = ["set", "group", "instance", "n", "m", "W", "ref", "ref_kind",
            lab, lab + " late jobs", "WSPT", "best_of_17", "best_of_17_name", "seed", "GA_joint",
            lab + " gap%", "WSPT gap%", "best_of_17 gap%", "seed gap%", "GA_joint gap%"]
    out = df[cols].rename(columns={lab: "LLM lost", lab + " late jobs": "LLM late jobs",
                                   lab + " gap%": "LLM gap%"})
    out.round(4).to_csv(os.path.join(d, "per_instance.csv"), index=False)


def write_readme(info, S):
    a = S[S.group == "all"].set_index("set")
    sets = ["family", "family, GA's 24", "adversarial", "similarity m=1", "similarity m=4"]
    head = "| Method | " + " | ".join(f"{s} ({int(a.loc[s, 'instances'])})" for s in sets) + " |"
    lines = ["# LLM-generated heuristics: outputs (report 14)\n",
             "One folder per model. A heuristic is one rule, written once by the LLM and then "
             "used unchanged on every instance.\n",
             "- `heuristic.py`: the heuristic that was tested, with the model's own idea\n"
             "- `all_heuristics.md`: every heuristic the model wrote in its run\n"
             "- `per_instance.csv`: weight lost on every instance, next to WSPT, the best of "
             "the 17 rules on that instance, the WSPT seed and GA-joint\n",
             "Top level: `per_instance.csv` (all methods), `summary_by_set.csv`.\n",
             "\n## Mean gap (%) per instance set\n",
             "Family: gap to the LP bound, averaged over the 12 families as in report 01. "
             "GA-joint was run on only 24 of the 102 family instances (2 per family), so the "
             "second column repeats the comparison on those 24. Adversarial and similarity: "
             "gap to the exact optimum.\n\n",
             head, "\n|---|" + "---|" * len(sets) + "\n"]
    for k, lab in [("WSPT", "WSPT"), ("best_of_17", "best of 17 rules (per instance)"),
                   ("seed", "WSPT seed, new decoder")] + \
            [(m["label"], f"**{m['label']}**") for m in info] + [("GA_joint", "GA-joint")]:
        lines.append(f"| {lab} | " + " | ".join(f"{a.loc[s, k]:.2f}" for s in sets) + " |\n")
    lines.append("\n## Models\n\n| Folder | Called through | Written | Ran | Tested heuristic's idea |\n"
                 "|---|---|---|---|---|\n")
    for m in info:
        lines.append(f"| `{FOLDER[m['model']]}/` | {m['how']} | {m['written']} | {m['working']} "
                     f"| {m['best']['idea']} |\n")
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(lines))


if __name__ == "__main__":
    main()
