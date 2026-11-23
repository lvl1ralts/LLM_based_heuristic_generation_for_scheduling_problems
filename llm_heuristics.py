#!/usr/bin/env python3
"""
LLM-generated heuristics (EoH-style demo) for Pm | r_i = 0 | sum_i w_i U_i.

  python llm_heuristics.py --list-models            # what the key in .env can call
  python llm_heuristics.py --mock                   # whole pipeline, no API calls
  python llm_heuristics.py                          # model from .env
  python llm_heuristics.py --models gemini-2.5-flash,gemini-2.5-pro
  python llm_heuristics.py --models grok-4-fast      # xAI, key XAI_API_KEY in .env
  python llm_heuristics.py --models groq:openai/gpt-oss-120b   # Groq, key GROQ_API_KEY

The GA study showed that a rule needs two parts: which jobs to try first
(priority) and which machine each job goes on (placement). The GA found good
schedules but gives no rule a person can read. Here an LLM writes both parts as
short Python functions, and an Evolution-of-Heuristics loop (Liu et al., ICML
2024) improves them:

  priority(f, c)              -> one score per job, higher = try first
  choose_machine(i, f, c, M)  -> one score per machine for job i

The decoder is the GA's assignment decoder: jobs are taken in priority order,
the machines where job i still fits (every job on that machine on time when run
in deadline order, which is exact for one machine) are offered, and the job
goes on the feasible machine with the highest score, or is late if none fits.
So generated code can only choose; it can never produce an infeasible schedule.

EoH loop: an initial population is written from scratch (operator i1); each
generation applies e1 (a new idea unlike the parents), e2 (a new idea from what
the parents share), m1 (change a parent's logic) and m2 (change a parent's
constants). Fitness is the mean % gap to the LP bound on 12 training instances
built from the tuning seeds (0-17), so no test instance is ever seen.

The best heuristic of each run is then tested on the instances the GA report
used, against the same references, so the numbers sit next to report 10's:
  adversarial   24 MetaOpt instances, exact optimum
  similarity    5 per case x 8 cases, m=1 (DP) and m=4 (MILP)
  benchmark     first 2 instances of each of the 12 families, LP bound (DP on m=1)

Writes to results/llm/<model>/:
  candidates.jsonl   every heuristic the LLM wrote: operator, idea, code, fitness
  generations.csv    best and mean fitness per generation
  best_heuristic.py  the winning code
  test.csv           best heuristic vs WSPT, best rule and GA-joint per instance
and results/llm/summary.csv across all models run.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
import urllib.error
import urllib.request
from bisect import bisect_right

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduling.environment import context, evaluate, features, lower_bound
from scheduling.heuristics import RULES
from scheduling.instances import P1Instance, gen_p1

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "llm")
API = "https://generativelanguage.googleapis.com/v1beta"
EPS = 1e-9


# --------------------------------------------------------------------------- #
# .env and the Gemini REST API (standard library only)
# --------------------------------------------------------------------------- #
def load_env(path=os.path.join(HERE, ".env")):
    env = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _request(url, key, body=None, timeout=300):
    req = urllib.request.Request(url, method="POST" if body else "GET",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json",
                                          "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def list_models(key):
    out, token = [], ""
    while True:
        J = _request(f"{API}/models?pageSize=200" + (f"&pageToken={token}" if token else ""), key)
        out += [m for m in J.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])]
        token = J.get("nextPageToken")
        if not token:
            return out


class Gemini:
    def __init__(self, model, key, temperature=1.0):
        self.model, self.key, self.temperature = model, key, temperature
        self.calls = self.failed = self.tokens_in = self.tokens_out = 0
        self.seconds = 0.0

    def __call__(self, prompt):
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": self.temperature}}
        url = f"{API}/models/{self.model}:generateContent"
        for attempt in range(6):
            t0 = time.time()
            try:
                J = _request(url, self.key, body)
            except urllib.error.HTTPError as e:
                raw = e.read().decode(errors="replace")
                msg = raw[:300]
                delay = re.search(r'"retryDelay":\s*"(\d+)', raw)
                if e.code == 429 and delay and int(delay.group(1)) > 300:
                    raise RuntimeError(f"quota used up for {self.model} (resets in "
                                       f"{int(delay.group(1)) // 3600} h); a free-tier key "
                                       f"may have no quota for this model") from None
                if e.code in (429, 500, 503, 504) and attempt < 5:
                    wait = int(delay.group(1)) + 2 if delay else 15 * 2 ** attempt
                    print(f"    [{self.model}] HTTP {e.code}, retrying in {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                if e.code == 404:
                    msg += "\n  (model not found: run with --list-models)"
                raise RuntimeError(f"HTTP {e.code}: {msg}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 5:
                    time.sleep(10)
                    continue
                raise RuntimeError(str(e)) from None
            self.seconds += time.time() - t0
            self.calls += 1
            u = J.get("usageMetadata", {})
            self.tokens_in += u.get("promptTokenCount", 0)
            self.tokens_out += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
            parts = (J.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts if not p.get("thought"))
        raise RuntimeError("no response")


XAI = "https://api.x.ai/v1"
GROQ = "https://api.groq.com/openai/v1"     # Groq (open models), not xAI's Grok


def _xai_request(path, key, body=None, timeout=300, base=XAI):
    req = urllib.request.Request(f"{base}/{path}", method="POST" if body else "GET",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}",
                                          "User-Agent": "llm-heuristics-demo/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def list_grok_models(key, base=XAI):
    return sorted(m["id"] for m in _xai_request("models", key, base=base).get("data", []))


class Grok:
    """OpenAI-format chat-completions API (xAI's Grok, or Groq with base=GROQ);
    same interface as Gemini. model is used for the results folder, so a Groq id
    like 'openai/gpt-oss-120b' is stored as 'groq_openai_gpt-oss-120b'."""

    def __init__(self, model, key, temperature=1.0, base=XAI):
        self.api_model, self.key, self.temperature, self.base = model, key, temperature, base
        self.model = ("groq_" if base == GROQ else "") + model.replace("/", "_")
        self.calls = self.failed = self.tokens_in = self.tokens_out = 0
        self.seconds = 0.0

    def __call__(self, prompt):
        body = {"model": self.api_model, "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}]}
        for attempt in range(6):
            t0 = time.time()
            try:
                J = _xai_request("chat/completions", self.key, body, base=self.base)
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if e.code in (401, 403):
                    raise RuntimeError(f"HTTP {e.code} for {self.model}: key rejected or no "
                                       f"credits on the xAI account: {msg}") from None
                if e.code == 429 and re.search(r"credit|spend|exhaust|quota|per day", msg, re.I):
                    raise RuntimeError(f"quota or credits used up for {self.model}: {msg}") from None
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    ra = e.headers.get("retry-after") if e.headers else None
                    wait = int(float(ra)) + 1 if ra and float(ra) <= 300 else 15 * 2 ** attempt
                    print(f"    [{self.model}] HTTP {e.code}, retrying in {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                if e.code in (400, 404) and "model" in msg.lower():
                    msg += "\n  (model not available: run with --list-models)"
                raise RuntimeError(f"HTTP {e.code}: {msg}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 5:
                    time.sleep(10)
                    continue
                raise RuntimeError(str(e)) from None
            self.seconds += time.time() - t0
            self.calls += 1
            u = J.get("usage", {})
            self.tokens_in += u.get("prompt_tokens", 0)
            self.tokens_out += u.get("completion_tokens", 0) + \
                (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            return ((J.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        raise RuntimeError("no response")


# --------------------------------------------------------------------------- #
# the decoder that runs LLM-written code
# --------------------------------------------------------------------------- #
def decode(inst, prio, choose):
    """Jobs in decreasing priority; job i goes on the feasible machine with the
    highest choose_machine score, or is late. Returns (weight lost, machine per job)."""
    p, d, w, m, n = inst.p, inst.d, inst.w, inst.m, inst.n
    f, c = Fields(features(inst)), Fields(context(inst))
    big = float(c["max_d"])
    dl = [[] for _ in range(m)]          # deadlines on each machine, EDD order
    cl = [[] for _ in range(m)]          # completion times, same order
    sl = [[] for _ in range(m)]          # slack d - C, same order
    load = np.zeros(m)
    mach = -np.ones(n, dtype=int)
    for i in np.argsort(-prio, kind="stable"):
        pi, di = float(p[i]), float(d[i])
        fits = np.zeros(m, dtype=bool)
        after = np.full(m, -1.0)
        finish = np.full(m, -1.0)
        pos = np.zeros(m, dtype=int)
        for j in range(m):
            k = bisect_right(dl[j], di)
            start = cl[j][k - 1] if k else 0.0
            if start + pi > di + EPS:
                continue
            later = min(sl[j][k:]) - pi if k < len(sl[j]) else big
            if later < -EPS:
                continue
            earlier = min(sl[j][:k]) if k else big
            fits[j], pos[j], finish[j] = True, k, start + pi
            after[j] = min(earlier, later, di - start - pi)
        if not fits.any():
            continue
        if fits.sum() == 1:
            j = int(np.argmax(fits))
        else:
            M = Fields(load=load.copy(), count=np.array([len(x) for x in dl], float),
                     min_slack=np.array([min(x) if x else big for x in sl]),
                     fits=fits.copy(), slack_after=after.copy(), finish_after=finish.copy())
            s = np.asarray(choose(int(i), f, c, M), dtype=float).ravel()
            if s.shape[0] != m:
                s = np.resize(s, m)
            s = np.where(np.isfinite(s), s, -1e300)
            s[~fits] = -np.inf
            j = int(np.argmax(s))
        k = pos[j]
        dl[j].insert(k, di)
        cl[j].insert(k, (cl[j][k - 1] if k else 0.0) + pi)
        sl[j].insert(k, 0.0)
        for s_ in range(k + 1, len(cl[j])):
            cl[j][s_] += pi
        sl[j] = [a - b for a, b in zip(dl[j], cl[j])]
        load[j] += pi
        mach[i] = j
    for j in range(m):                   # every machine, in EDD order, is on time
        idx = sorted(np.where(mach == j)[0], key=lambda i: d[i])
        assert np.all(np.cumsum(p[idx]) <= d[idx] + 1e-6), "late job on a machine"
    return float(w[mach < 0].sum()), mach


# --------------------------------------------------------------------------- #
# checking and running generated code
# --------------------------------------------------------------------------- #
BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "globals", "locals",
                "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "exit",
                "quit", "help", "memoryview", "type", "object", "super", "classmethod"}
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in ["abs", "min", "max", "sum", "len", "range", "enumerate", "zip",
                           "float", "int", "bool", "list", "tuple", "dict", "sorted",
                           "reversed", "round", "pow", "any", "all", "map", "filter",
                           "isinstance", "ValueError", "ZeroDivisionError", "Exception"]}


def check_code(code):
    """Static check: only plain functions using np / math, no imports, no dunder access."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                             ast.With, ast.AsyncWith, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Await, ast.Yield, ast.YieldFrom)):
            raise ValueError(f"not allowed: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"not allowed: attribute {node.attr}")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in BANNED_CALLS):
            raise ValueError(f"not allowed: name {node.id}")
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "np" and node.attr in {
                    "load", "save", "savetxt", "loadtxt", "fromfile", "tofile", "memmap",
                    "genfromtxt", "ctypeslib", "lib", "testing", "f2py", "random"}):
            raise ValueError(f"not allowed: np.{node.attr}")
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for need in ("priority", "choose_machine"):
        if need not in names:
            raise ValueError(f"missing function {need}")


class Fields(dict):
    """dict that also allows f.wp, since the models write it both ways."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


def compile_code(code):
    check_code(code)
    g = {"np": np, "math": math, "__builtins__": SAFE_BUILTINS}
    exec(compile(code, "<llm>", "exec"), g)
    return g["priority"], g["choose_machine"]


def run_code(code, data):
    """Weight lost on each instance in `data` ([(p, d, w, m), ...])."""
    prio_fn, choose = compile_code(code)
    lost = []
    for p, d, w, m in data:
        inst = P1Instance(p=p, d=d, w=w, m=m)
        f, c = Fields(features(inst)), Fields(context(inst))
        prio = np.asarray(prio_fn(f, c), dtype=float).ravel()
        if prio.shape[0] != inst.n:
            raise ValueError(f"priority returned {prio.shape[0]} scores for {inst.n} jobs")
        prio = np.nan_to_num(prio, nan=0.0, posinf=1e12, neginf=-1e12)
        lost.append(decode(inst, prio, choose)[0])
    return lost


def _worker(code, data, q):
    try:
        q.put(("ok", run_code(code, data)))
    except Exception as e:                      # report any failure of generated code
        q.put(("error", f"{type(e).__name__}: {e}"[:400]))


def run_isolated(code, data, timeout):
    """run_code in a separate process, killed after `timeout` seconds."""
    q = mp.get_context("spawn").Queue()
    proc = mp.get_context("spawn").Process(target=_worker, args=(code, data, q), daemon=True)
    proc.start()
    try:
        status, val = q.get(timeout=timeout)
    except Exception:
        status, val = "error", f"timed out after {timeout}s"
    proc.join(1)
    if proc.is_alive():
        proc.kill()
    return status, val


# --------------------------------------------------------------------------- #
# instances
# --------------------------------------------------------------------------- #
def _data(insts):
    return [(i.p, i.d, i.w, i.m) for i in insts]


def train_set():
    """12 instances from the tuning seeds (0-17), never used in any test."""
    spec = [dict(m=2, T=0.3), dict(m=2, T=0.6), dict(m=3, T=0.5, wm="correlated"),
            dict(m=4, T=0.2), dict(m=4, T=0.5), dict(m=4, T=0.7, R=0.3),
            dict(m=4, T=0.5, pm="bimodal"), dict(m=4, T=0.5, wm="anticorrelated"),
            dict(m=5, T=0.5, pm="lognormal"), dict(m=6, T=0.4, R=0.1),
            dict(m=8, T=0.5), dict(m=1, T=0.5)]
    out = []
    for s, sp in enumerate(spec):
        out.append(gen_p1(n=50, m=sp["m"], seed=s, tardiness_factor=sp["T"],
                          due_range=sp.get("R", 0.6), weight_model=sp.get("wm", "uncorrelated"),
                          p_model=sp.get("pm", "uniform")))
    return out


def test_sets():
    """The GA report's instances with its references and baseline results."""
    import glob
    from similarity_cases import gen_case
    from scheduling.suites import all_suites
    G = os.path.join(HERE, "results", "ga")
    sets = {}

    rows, insts = [], []
    ga = pd.read_csv(os.path.join(G, "adversarial.csv")).set_index("instance")
    for path in sorted(glob.glob(os.path.join(HERE, "results", "metaopt", "*.json"))):
        J = json.load(open(path))
        name = os.path.basename(path)[:-5]
        insts.append(P1Instance(p=J["p"], d=J["d"], w=J["w"], m=J["m"], name=name))
        rows.append(dict(group="adversarial", instance=name, ref=J["opt_lost"], **_base(ga.loc[name])))
    sets["adversarial"] = (insts, rows)

    sim = pd.read_csv(os.path.join(G, "similarity.csv"))
    for m_val, label in [(1, "similarity m=1"), (4, "similarity m=4")]:
        rows, insts = [], []
        for _, r in sim[sim.m == m_val].iterrows():
            tags = r.case.replace(" similar", "").split(" & ") if r.case != "none similar" else []
            inst = gen_case(int(r.n), m_val, int(r.instance.split("_s")[1]),
                            "p" in tags, "d" in tags, "w" in tags)
            assert abs(inst.total_weight - r.W) < 1e-6, "similarity instance mismatch"
            insts.append(inst)
            rows.append(dict(group=label, instance=f"{r.case}|{r.instance}", ref=r.opt, **_base(r)))
        sets[label] = (insts, rows)

    bench = pd.read_csv(os.path.join(G, "benchmark.csv"))
    by_name = {i.name: i for fam in all_suites().values() for i in fam}
    rows, insts = [], []
    for _, r in bench.iterrows():
        insts.append(by_name[r.instance])
        rows.append(dict(group="benchmark", instance=f"{r.family}|{r.instance}", ref=r.ref,
                         **_base(r)))
    sets["benchmark"] = (insts, rows)
    return sets


def _base(r):
    return dict(W=float(r.W), WSPT=float(r.WSPT), best_rule=float(r.best_rule),
                GA_joint=float(r["GA-joint"]))


def gap(lost, ref, W):
    return 100.0 * (lost - ref) / max(W - ref, EPS)


# --------------------------------------------------------------------------- #
# prompts (EoH operators)
# --------------------------------------------------------------------------- #
TASK = """We schedule n jobs on m identical machines. Every job is available at time 0
and has a processing time p, a deadline d and a weight w. A job is on time if it
finishes by its deadline; a late job earns nothing, so it can be left out. The
goal is to keep as much total weight on time as possible (equivalently, to
minimise the total weight of late jobs).

You design a heuristic made of two Python functions.

1) priority(f, c) -> numpy array of n scores. Jobs are tried in decreasing score.
2) choose_machine(i, f, c, M) -> numpy array of m scores. Job i is offered only
   to the machines where it still fits (every job on that machine stays on time
   when the machine runs its jobs in deadline order). The job goes on the
   feasible machine with the highest score. If no machine fits, the job is late.
   This function is called only when two or more machines fit.

f holds per-job numpy arrays (length n):
  p, d, w; wp = w/p; slack = d - p; laxity = (d - p)/d; urgency = p/d;
  dens = w/(p*d); headroom = m*d - (total p of jobs due no later than d);
  headroom_norm = headroom / total p; d_rank (0..1 by deadline);
  p_norm, w_norm, d_norm (divided by their max); cum_p_before (total p due no
  later than d).
c holds instance scalars: m, n, total_p, total_w, mean_p, max_d, min_d, mean_d,
  load = total_p / (m * max_d).
M holds per-machine numpy arrays (length m) at the moment job i is placed:
  load (total processing time already on the machine), count (jobs on it),
  min_slack (smallest d - finish time among its jobs; max_d if empty),
  fits (bool), slack_after (smallest slack on the machine if job i is added;
  -1 where it does not fit), finish_after (finish time of job i there; -1 where
  it does not fit).

f, c and M are dictionaries: write f["wp"], c["m"], M["slack_after"].

Rules: use only numpy (as np) and math, which are already available; no
import statements; no randomness; return arrays of the right length; keep the
code short and readable so a person can understand the rule."""

FORMAT = """Reply with your idea in one sentence inside curly braces {}, then the code
for both functions in a single ```python code block. No other text."""


def _show(parents):
    return "\n\n".join(f"Heuristic {k + 1} (mean gap {h['fitness']:.3f}%, lower is better)\n"
                       f"Idea: {h['idea']}\n```python\n{h['code']}\n```"
                       for k, h in enumerate(parents))


def prompt(op, parents):
    if op == "i1":
        return f"{TASK}\n\nDesign a new heuristic.\n{FORMAT}"
    if op == "e1":
        return (f"{TASK}\n\nHere are existing heuristics:\n\n{_show(parents)}\n\n"
                f"Design a new heuristic whose idea is clearly different from all of "
                f"these.\n{FORMAT}")
    if op == "e2":
        return (f"{TASK}\n\nHere are existing heuristics:\n\n{_show(parents)}\n\n"
                f"Find the idea these heuristics share, then design a new heuristic "
                f"built on that idea but different from each of them.\n{FORMAT}")
    if op == "m1":
        return (f"{TASK}\n\nHere is a heuristic:\n\n{_show(parents)}\n\n"
                f"Change its logic to make it better. Keep what works.\n{FORMAT}")
    if op == "m2":
        return (f"{TASK}\n\nHere is a heuristic:\n\n{_show(parents)}\n\n"
                f"Keep its logic but change the constants or weights in it to make "
                f"it better.\n{FORMAT}")
    raise ValueError(op)


def parse(text):
    idea = re.search(r"\{(.*?)\}", text, re.S)
    code = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if not code:
        raise ValueError("no code block in the reply")
    return (idea.group(1).strip() if idea else "").replace("\n", " "), code.group(1).strip()


# --------------------------------------------------------------------------- #
# offline stand-in for the API, to test the loop without a key
# --------------------------------------------------------------------------- #
class MockLLM:
    IDEAS = [
        ("value per unit time, placed where the most slack is left",
         "return f['wp']", "return M['slack_after']"),
        ("value per unit time with a deadline tie-break, best fit placement",
         "return f['wp'] - {a} * f['d_norm']", "return -M['slack_after']"),
        ("weight density, least loaded machine", "return f['dens']", "return -M['load']"),
        ("value per time scaled by headroom, earliest finish",
         "return f['wp'] * (1 + {a} * np.clip(f['headroom_norm'], 0, 1))",
         "return -M['finish_after']"),
    ]

    def __init__(self):
        self.calls = self.failed = self.tokens_in = self.tokens_out = 0
        self.seconds, self.k = 0.0, 0
        self.model = "mock"

    def __call__(self, prompt_text):
        self.calls += 1
        idea, pr, ch = self.IDEAS[self.k % len(self.IDEAS)]
        a = round(0.001 * (1 + self.k), 4)
        self.k += 1
        if self.k == 3:                      # one broken reply, to exercise the error path
            return "{broken} ```python\ndef priority(f, c):\n    import os\n```"
        return (f"{{{idea}}}\n```python\ndef priority(f, c):\n    {pr.format(a=a)}\n\n"
                f"def choose_machine(i, f, c, M):\n    {ch}\n```")


# --------------------------------------------------------------------------- #
# the EoH loop
# --------------------------------------------------------------------------- #
def evolve(llm, train, refs, args, log):
    data = _data(train)
    W = [i.total_weight for i in train]
    pop, seen, gens = [], set(), []

    def fitness(code):
        status, val = run_isolated(code, data, args.timeout)
        if status != "ok":
            return None, val
        return float(np.mean([gap(l, r, w) for l, r, w in zip(val, refs, W)])), None

    out_of_quota = []

    def new(op, parents, gen, given=None):
        if out_of_quota or (given is None and llm.calls >= args.max_calls):
            return
        try:
            text = given or llm(prompt(op, parents))
            idea, code = parse(text)
            if code in seen:
                raise ValueError("duplicate of an earlier heuristic")
            seen.add(code)
            fit, err = fitness(code)
        except Exception as e:
            idea, code, fit, err = "", "", None, f"{type(e).__name__}: {e}"[:400]
            if "quota used up" in str(e):
                out_of_quota.append(str(e))
        h = dict(gen=gen, op=op, parents=[p["id"] for p in parents], idea=idea, code=code,
                 fitness=fit, error=err, id=len(log))
        log.append(h)
        print(f"    gen {gen} {op}: " + (f"gap {fit:6.3f}%  {idea[:70]}" if fit is not None
                                         else f"failed  {err[:90]}"), flush=True)
        if fit is not None:
            pop.append(h)
        else:
            llm.failed += 1

    def select(k):
        ranked = sorted(pop, key=lambda h: h["fitness"])
        pr = 1.0 / (np.arange(len(ranked)) + 1 + len(ranked))
        idx = rng.choice(len(ranked), size=min(k, len(ranked)), replace=False, p=pr / pr.sum())
        return [ranked[i] for i in idx]

    rng = np.random.default_rng(args.seed)
    new("seed", [], 0, SEED)                   # the existing rule, ported to this decoder
    for _ in range(args.pop * 2):              # rest of the population, written from scratch
        if len(pop) >= args.pop:
            break
        new("i1", [], 0)
    if not pop:
        raise RuntimeError("the LLM produced no working heuristic")
    pop[:] = sorted(pop, key=lambda h: h["fitness"])[:args.pop]
    gens.append(dict(gen=0, best=pop[0]["fitness"], mean=np.mean([h["fitness"] for h in pop]),
                     calls=llm.calls))
    for g in range(1, args.gens + 1):
        if out_of_quota or llm.calls >= args.max_calls:
            print(f"  stopping after generation {g - 1}: "
                  + (out_of_quota[0] if out_of_quota else f"{llm.calls} calls (--max-calls)"))
            break
        for op in ["e1", "e2", "m1", "m2"]:
            for _ in range(args.per_op):
                new(op, select(2 if op.startswith("e") else 1), g)
        uniq = {}
        for h in sorted(pop, key=lambda h: h["fitness"]):
            uniq.setdefault(round(h["fitness"], 9), h)   # one per distinct score
        pop[:] = list(uniq.values())[:args.pop]
        gens.append(dict(gen=g, best=pop[0]["fitness"], mean=np.mean([h["fitness"] for h in pop]),
                         calls=llm.calls))
        print(f"  generation {g}: best {pop[0]['fitness']:.3f}%  ({llm.calls} calls)", flush=True)
    return pop[0], pd.DataFrame(gens)


# the starting point every run gets: WSPT as before, first machine that fits
SEED = """{WSPT priority (weight per unit of processing time); the job goes on the first machine where it fits.}
```python
def priority(f, c):
    return f["wp"]


def choose_machine(i, f, c, M):
    return -np.arange(len(M["fits"]))
```"""
# a second hand-written reference, not given to the LLM: pack each job where it fits most tightly
BEST_FIT = SEED.split("```python")[1].split("```")[0].replace(
    'return -np.arange(len(M["fits"]))', 'return -M["slack_after"]')


TEST_COLS = ["WSPT", "best_rule", "seed", "WSPT_bestfit", "LLM", "GA_joint"]


def test(best_code, sets, timeout):
    """Best LLM heuristic and the two hand-written references on every test set.
    WSPT, best_rule and GA_joint come from the GA report's CSVs."""
    codes = dict(seed=parse(SEED)[1], WSPT_bestfit=BEST_FIT, LLM=best_code)
    rows = []
    for name, (insts, base) in sets.items():
        lost = {}
        for k, code in codes.items():
            status, lost[k] = run_isolated(code, _data(insts), timeout)
            if status != "ok":
                raise RuntimeError(f"{k} failed on {name}: {lost[k]}")
        for t, b in enumerate(base):
            r = dict(b, **{k: lost[k][t] for k in codes})
            for k in TEST_COLS:
                r[k + " gap%"] = gap(r[k], r["ref"], r["W"])
            rows.append(r)
    return pd.DataFrame(rows)


def run_model(llm, train, refs, wspt_train, sets, args):
    out = os.path.join(OUT, llm.model)
    os.makedirs(out, exist_ok=True)
    print(f"\n== {llm.model} ==", flush=True)
    log, t0 = [], time.time()
    try:
        best, gens = evolve(llm, train, refs, args, log)
        if llm.calls == 0:
            raise RuntimeError("no API call went through, nothing to report")
    finally:
        with open(os.path.join(out, "candidates.jsonl"), "w", encoding="utf-8") as fh:
            for h in log:
                fh.write(json.dumps(h) + "\n")
    gens["WSPT train gap%"] = wspt_train
    gens.to_csv(os.path.join(out, "generations.csv"), index=False)
    with open(os.path.join(out, "best_heuristic.py"), "w", encoding="utf-8") as fh:
        fh.write(f'"""{llm.model}, generation {best["gen"]} ({best["op"]}), '
                 f'training gap {best["fitness"]:.3f}%\n\n{best["idea"]}\n"""\n'
                 f"import math\n\nimport numpy as np\n\n\n{best['code']}\n")
    T = test(best["code"], sets, args.timeout * 10)
    T.to_csv(os.path.join(out, "test.csv"), index=False)
    seed_fit = next(h["fitness"] for h in log if h["op"] == "seed")
    row = dict(model=llm.model, calls=llm.calls, failed=llm.failed,
               working=sum(h["fitness"] is not None for h in log if h["op"] != "seed"),
               tokens_in=llm.tokens_in, tokens_out=llm.tokens_out,
               api_minutes=round(llm.seconds / 60, 1), total_minutes=round((time.time() - t0) / 60, 1),
               train_WSPT=wspt_train, train_seed=seed_fit, train_best=best["fitness"],
               best_from=f"gen {best['gen']} {best['op']}", best_idea=best["idea"])
    for grp, sub in T.groupby("group", sort=False):
        for k in TEST_COLS:
            row[f"{grp} | {k}"] = round(sub[k + " gap%"].mean(), 3)
    for g, s in T.groupby("group", sort=False):
        print(f"  test {g:15s} " + "  ".join(f"{k} {s[k + ' gap%'].mean():6.2f}"
                                            for k in TEST_COLS), flush=True)
    return row


def self_check():
    """decode() with a fixed machine preference must equal ga.decode_assign."""
    from ga import decode_assign
    rng = np.random.default_rng(3)
    for s in range(30):
        inst = gen_p1(n=30, m=int(rng.integers(1, 5)), seed=91000 + s,
                      tardiness_factor=float(rng.uniform(0.2, 0.8)))
        prio, pref = rng.random(inst.n), rng.integers(0, inst.m, inst.n)
        lost, _ = decode(inst, prio, lambda i, f, c, M: -((np.arange(inst.m) - pref[i]) % inst.m))
        kept, _ = decode_assign(np.argsort(-prio, kind="stable"), pref,
                                inst.p, inst.d, inst.w, inst.m)
        assert abs(lost - (inst.total_weight - kept)) < 1e-6, "decoder mismatch"
    print("  self-check passed (decoder matches the GA's assignment decoder)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated models; names starting with 'grok' "
                                     "use xAI (default: GEMINI_MODEL)")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--mock", action="store_true", help="offline stand-in, no API calls")
    ap.add_argument("--pop", type=int, default=6, help="population size")
    ap.add_argument("--gens", type=int, default=5, help="generations")
    ap.add_argument("--per-op", type=int, default=1, help="offspring per operator per generation")
    ap.add_argument("--max-calls", type=int, default=18,
                    help="API calls per model (a free-tier key allows about 20 a day)")
    ap.add_argument("--timeout", type=int, default=60, help="seconds per training evaluation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):  # models write unicode (e.g. U+2011) in their ideas
        s.reconfigure(encoding="utf-8", errors="replace")

    env = load_env()
    key = os.environ.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEY", "")
    xkey = os.environ.get("XAI_API_KEY") or env.get("XAI_API_KEY", "")
    gkey = os.environ.get("GROQ_API_KEY") or env.get("GROQ_API_KEY", "")
    if args.list_models:
        if gkey:
            print("Groq (use as groq:<id>):")
            for m in list_grok_models(gkey, base=GROQ):
                print(f"  {m}")
        if key:
            print("Gemini:")
            for m in list_models(key):
                print(f"  {m['name'].split('/')[-1]:45s} {m.get('displayName', '')}")
        if xkey:
            print("Grok (xAI):")
            for m in list_grok_models(xkey):
                print(f"  {m}")
        return
    if args.mock:
        names = []
    else:
        names = [n.strip() for n in (args.models or env.get("GEMINI_MODEL", "gemini-3.8-flash"))
                 .split(",") if n.strip()]
        if any(n.startswith("grok") for n in names) and not xkey:
            sys.exit("No XAI_API_KEY in .env")
        if any(n.startswith("groq:") for n in names) and not gkey:
            sys.exit("No GROQ_API_KEY in .env")
        if any(n.startswith("gemini") or n.startswith("gemma") for n in names) and not key:
            sys.exit("No GEMINI_API_KEY in .env")

    self_check()
    train = train_set()
    refs = [lower_bound(i) for i in train]
    wspt_train = float(np.mean([gap(evaluate(i, RULES["WSPT"].fn), r, i.total_weight)
                                for i, r in zip(train, refs)]))
    print(f"  training: {len(train)} instances, WSPT mean gap {wspt_train:.3f}%")
    sets = test_sets()
    print("  test: " + ", ".join(f"{k} ({len(v[0])})" for k, v in sets.items()))

    if args.mock:
        llms = [MockLLM()]
    else:
        llms = [Grok(n[5:], gkey, base=GROQ) if n.startswith("groq:")
                else Grok(n, xkey) if n.startswith("grok") else Gemini(n, key) for n in names]
    import glob
    for llm in llms:
        try:
            row = run_model(llm, train, refs, wspt_train, sets, args)
        except RuntimeError as e:
            print(f"  {llm.model} stopped: {e}", flush=True)
            continue
        with open(os.path.join(OUT, llm.model, "summary.json"), "w") as fh:
            json.dump(row, fh, indent=1)
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*", "summary.json")))]
    if rows:                             # one row per model run so far, any process
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
