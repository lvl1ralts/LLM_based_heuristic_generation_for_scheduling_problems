#!/usr/bin/env python3
"""
BTP demo: an LLM writes a scheduling heuristic, and we run it on one instance.

  ..\\Code\\.venv\\Scripts\\python demo.py                        live call, Groq (GPT-OSS 120B)
  ..\\Code\\.venv\\Scripts\\python demo.py --model groq:qwen/qwen3.8-27b
  ..\\Code\\.venv\\Scripts\\python demo.py --model gemini-3.8-flash      (slow: ~1-2 min)
  ..\\Code\\.venv\\Scripts\\python demo.py --saved                no API: Gemma's rule from the report

What it does, step by step (one LLM request):
  1. shows the instance (6 jobs, 2 machines, from MetaOpt) and its optimum
  2. runs WSPT on it, the way every classical rule schedules
  3. sends the LLM the same request that produced Gemma's rule in the report:
     the task description, an earlier heuristic, and "change its logic to make it better"
  4. checks the code it returns (no imports, no file access) and runs it in a separate
     process on the instance, and on the 12 training instances for a training gap
  5. prints the schedule and the weight lost next to WSPT and the optimum

Everything is saved in output/: the full prompt, the raw reply, and the heuristic as a
.py file. API keys are read from ../Code/.env and never printed. If the call fails
(no network, quota), the demo says so and falls back to Gemma's saved heuristic.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "Code")
sys.path.insert(0, CODE)

import numpy as np

import llm_heuristics as L
from scheduling.environment import accepted_mask, evaluate, lower_bound, schedule_of
from scheduling.heuristics import RULES
from scheduling.instances import P1Instance

OUT = os.path.join(HERE, "output")
LINE = "-" * 72

# The earlier heuristic shown to the model: WSPT + "machine with the most slack left".
# This is the exact parent from the report's Gemma run (training gap 4.147%).
PARENT = dict(fitness=4.147319305789998,
              idea="Use WSPT priority to focus on high-weight, short-processing jobs, and "
                   "assign them to the machine that maximizes the resulting minimum slack "
                   "to preserve future flexibility.",
              code='def priority(f, c):\n    return f["wp"]\n\n\n'
                   'def choose_machine(i, f, c, M):\n    return M["slack_after"]')

# Gemma 4 31B's reply to that request in the report (used with --saved or as a fallback)
GEMMA = dict(idea="Use WSPT priority to focus on high-value jobs and a Best-Fit machine "
                  "selection strategy by maximizing the load on the chosen machine to "
                  "preserve large contiguous time blocks on other machines.",
             code='def priority(f, c):\n    return f["wp"]\n\n\n'
                  'def choose_machine(i, f, c, M):\n    return M["load"]')


def say(title):
    print(f"\n{LINE}\n {title}\n{LINE}", flush=True)


def jobname(i):
    return f"J{i + 1}"


def show_schedule(rows, m, late):
    """rows: (job, machine, start, finish)."""
    for j in range(m):
        parts = [f"{jobname(i)}[{s:g}-{e:g}]" for i, mj, s, e in sorted(rows, key=lambda r: r[2])
                 if mj == j]
        print(f"   M{j + 1}: " + ("  ".join(parts) if parts else "(empty)"))
    print(f"   late: {', '.join(jobname(i) for i in late) if late else 'none'}")


def machine_rows(inst, mach):
    rows = []
    for j in range(inst.m):
        t = 0.0
        for i in sorted(np.where(mach == j)[0], key=lambda i: inst.d[i]):
            rows.append((int(i), j, t, t + float(inst.p[i])))
            t += float(inst.p[i])
    return rows


# --------------------------------------------------------------------------- #
# running LLM code in a separate process (it is generated code)
# --------------------------------------------------------------------------- #
def _worker(code, data, q):
    try:
        prio_fn, choose = L.compile_code(code)
        res = []
        for p, d, w, m in data:
            inst = P1Instance(p=p, d=d, w=w, m=m)
            f, c = L.Fields(L.features(inst)), L.Fields(L.context(inst))
            prio = np.nan_to_num(np.asarray(prio_fn(f, c), dtype=float).ravel(),
                                 nan=0.0, posinf=1e12, neginf=-1e12)
            lost, mach = L.decode(inst, prio, choose)
            res.append((lost, mach.tolist()))
        q.put(("ok", res))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"[:300]))


def run_code(code, data, timeout=60):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(code, data, q), daemon=True)
    proc.start()
    try:
        status, val = q.get(timeout=timeout)
    except Exception:
        status, val = "error", f"timed out after {timeout}s"
    proc.join(1)
    if proc.is_alive():
        proc.kill()
    if status != "ok":
        raise RuntimeError(val)
    return val


# --------------------------------------------------------------------------- #
# one LLM request, with a hard time limit
# --------------------------------------------------------------------------- #
def ask(model, prompt_text, limit):
    env = L.load_env()
    if model.startswith("groq:"):
        key = env.get("GROQ_API_KEY", "")
        body = {"model": model[5:], "temperature": 1.0,
                "messages": [{"role": "user", "content": prompt_text}]}
        call = lambda: L._xai_request("chat/completions", key, body, timeout=limit, base=L.GROQ)
        read = lambda J: J["choices"][0]["message"]["content"]
    else:
        key = env.get("GEMINI_API_KEY", "")
        body = {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
                "generationConfig": {"temperature": 1.0}}
        call = lambda: L._request(f"{L.API}/models/{model}:generateContent", key, body,
                                  timeout=limit)
        read = lambda J: "".join(p.get("text", "") for p in
                                 J["candidates"][0]["content"]["parts"] if not p.get("thought"))
    if not key:
        raise RuntimeError("no API key for this model in Code/.env")
    box = {}

    def go():
        try:
            box["reply"] = read(call())
        except Exception as e:
            msg = e.read().decode(errors="replace")[:200] if hasattr(e, "read") else str(e)
            box["error"] = f"{type(e).__name__}: {msg}"

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t0 = time.time()
    while t.is_alive() and time.time() - t0 < limit:
        print(f"\r   waiting for {model} ... {time.time() - t0:4.0f}s", end="", flush=True)
        t.join(0.5)
    print()
    if "reply" in box:
        return box["reply"], time.time() - t0
    raise RuntimeError(box.get("error", f"no answer within {limit}s"))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="groq:openai/gpt-oss-120b",
                    help="groq:<id> for Groq, otherwise a Gemini API model id")
    ap.add_argument("--saved", action="store_true", help="skip the API, use Gemma's rule")
    ap.add_argument("--limit", type=int, default=120, help="seconds to wait for the LLM")
    args = ap.parse_args()
    for s in (sys.stdout, sys.stderr):    # model replies can contain unicode (e.g. U+2011)
        s.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(OUT, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    J = json.load(open(os.path.join(HERE, "instance.json")))
    inst = P1Instance(p=J["p"], d=J["d"], w=J["w"], m=J["m"])

    # 1. the instance ---------------------------------------------------------
    say(f"1. The instance: {J['name']}  ({inst.n} jobs, {inst.m} machines)")
    print("   job    " + "".join(f"{jobname(i):>6}" for i in range(inst.n)))
    for lab, v in [("p (len)", inst.p), ("d (due)", inst.d), ("w (val)", inst.w)]:
        print(f"   {lab:7s}" + "".join(f"{x:6g}" for x in v))
    print(f"   total weight {inst.total_weight:g};  optimum (HiGHS integer program) loses "
          f"{J['opt_lost']:g}")

    # 2. WSPT --------------------------------------------------------------------
    say("2. WSPT, the best classical rule (earliest-free machine)")
    keep = accepted_mask(inst, RULES["WSPT"].fn, RULES["WSPT"].decoder)
    rows = schedule_of(sorted((inst.d[i], inst.p[i], int(i)) for i in np.where(keep)[0]),
                       inst.m)
    show_schedule(rows, inst.m, [i for i in range(inst.n) if not keep[i]])
    wspt_lost = evaluate(inst, RULES["WSPT"].fn, RULES["WSPT"].decoder)
    print(f"   WSPT loses {wspt_lost:g}")

    # 3. ask the LLM ---------------------------------------------------------------
    prompt_text = L.prompt("m1", [PARENT])
    open(os.path.join(OUT, "prompt.txt"), "w", encoding="utf-8").write(prompt_text)
    source = "saved: Gemma 4 31B (report)"
    if args.saved:
        say("3. Heuristic from the report (no API call)")
        idea, code = GEMMA["idea"], GEMMA["code"]
    else:
        say(f"3. Asking the LLM ({args.model}) to improve an earlier heuristic")
        print("   prompt = task description (problem, the two functions, their inputs)")
        print("          + this earlier heuristic:")
        print("              priority:        WSPT, f[\"wp\"] = weight / processing time")
        print("              choose_machine:  machine with the most slack left (spreads jobs)")
        print("          + \"Change its logic to make it better. Keep what works.\"")
        print(f"   (full prompt: output/prompt.txt, {len(prompt_text)} characters)")
        idea = code = None
        for attempt in range(1, 4):               # a reply can be unusable: ask again
            try:
                reply, secs = ask(args.model, prompt_text, args.limit)
            except Exception as e:                # network / quota: no point retrying here
                print(f"   ! the call did not work ({str(e)[:150]})")
                break
            open(os.path.join(OUT, f"reply_{stamp}_{attempt}.txt"), "w",
                 encoding="utf-8").write(reply)
            try:
                idea, code = L.parse(reply)
                L.check_code(code)                # no imports, no file access, both functions
                source = f"live: {args.model}, answered in {secs:.0f}s"
                break
            except Exception as e:
                idea = code = None
                print(f"   reply {attempt} rejected by the safety check ({str(e)[:80]}); "
                      f"asking again")
        if code is None:
            print("   ! falling back to the heuristic Gemma wrote for the same request")
            idea, code = GEMMA["idea"], GEMMA["code"]

    # 4. the heuristic -----------------------------------------------------------
    say(f"4. The heuristic  [{source}]")
    print(f"   idea: {idea}\n")
    for ln in code.splitlines():
        print("      " + ln)
    path = os.path.join(OUT, f"heuristic_{stamp}.py")
    open(path, "w", encoding="utf-8").write(
        f'"""Heuristic for Pm | r=0 | sum w U  ({source})\n\nIdea: {idea}\n"""\n'
        f"import math\n\nimport numpy as np\n\n\n{code}\n")
    print(f"\n   saved to output/{os.path.basename(path)}")

    # 5. run it --------------------------------------------------------------------
    say("5. Running it on the instance (separate process, code checked first)")
    train = L.train_set()
    data = [(inst.p, inst.d, inst.w, inst.m)] + [(t.p, t.d, t.w, t.m) for t in train]
    try:
        res = run_code(code, data)
    except Exception as e:
        print(f"   ! the generated code failed: {e}")
        print("   ! (this happens; the loop would simply discard it) -> using Gemma's rule")
        code, idea, source = GEMMA["code"], GEMMA["idea"], "saved: Gemma 4 31B (report)"
        res = run_code(code, data)
    lost, mach = res[0][0], np.array(res[0][1])
    show_schedule(machine_rows(inst, mach), inst.m, [i for i in range(inst.n) if mach[i] < 0])

    refs = [lower_bound(t) for t in train]
    g = lambda l, r, t: L.gap(l, r, t.total_weight)
    llm_gap = np.mean([g(r[0], ref, t) for r, ref, t in zip(res[1:], refs, train)])
    wspt_gap = np.mean([g(evaluate(t, RULES["WSPT"].fn), ref, t) for ref, t in zip(refs, train)])

    # 6. the best rule from the full run, for comparison (our own saved code) ----
    live = source.startswith("live")
    if live:
        say("6. For comparison: the best rule from our full run (Gemma 4 31B)")
        for ln in GEMMA["code"].splitlines():
            if ln.strip():
                print("      " + ln)
        prio_fn, choose = L.compile_code(GEMMA["code"])
        f, c = L.Fields(L.features(inst)), L.Fields(L.context(inst))
        g_lost, g_mach = L.decode(inst, np.asarray(prio_fn(f, c), dtype=float), choose)
        show_schedule(machine_rows(inst, g_mach), inst.m,
                      [i for i in range(inst.n) if g_mach[i] < 0])

    say("Result")
    tag = lambda v: "   <- optimal" if v <= J["opt_lost"] + 1e-9 else ""
    row = lambda lab, v, t="": print(f"      {lab:<34}{v:>4g}{t}")
    print(f"   on this instance (weight lost, lower is better)")
    row("WSPT", wspt_lost)
    row("LLM rule just generated" if live else "Gemma's rule (from our run)", lost, tag(lost))
    if live:
        row("best rule from our run (Gemma)", g_lost, tag(g_lost))
    row("optimum", J["opt_lost"])
    print(f"   on the 12 training instances (mean gap to the LP bound, lower is better)")
    print(f"      WSPT {wspt_gap:.2f}%    " + ("rule just generated" if live else "Gemma's rule")
          + f" {llm_gap:.2f}%" + ("    Gemma's rule 3.47%" if live else ""))
    json.dump(dict(source=source, idea=idea, code=code, lost=lost, wspt_lost=wspt_lost,
                   opt_lost=J["opt_lost"], train_gap=llm_gap, train_gap_wspt=wspt_gap),
              open(os.path.join(OUT, f"result_{stamp}.json"), "w"), indent=1)
    print()


if __name__ == "__main__":
    main()
