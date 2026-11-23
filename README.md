# Code — Pm | r_i = 0 | Σ w_i U_i

Minimising the **total weight of late jobs** on identical parallel machines.

This folder holds the project code: the instance generator, the scheduling
environment, the priority rules, the benchmark across instance families, and
the adversarial-instance search.

## Layout

```
Code/
  scheduling/
    instances.py         instance representation + Potts-Van Wassenhove generator
    environment.py       feasibility oracle, features, the two decoders,
                         LP lower bound, exact single-machine DP
    heuristics.py        17 priority rules (9 from week 1, 8 added after)
    suites.py            12 held-out instance families (5 from week 1, 7 added after)
  run_benchmark.py       runs every rule on every family
  adversarial_search.py  searches for instances where WSPT fails
  metaopt.py             MetaOpt search: builds, per family, the instance where
                         WSPT / WSPT-Laxity lose most against the exact optimum
  report_metaopt.py      explains and plots those instances (results/metaopt/)
  similarity_cases.py    all rules on the eight similar/varied p, d, w cases
  ga.py                  genetic algorithm evolving priority + machine assignment
                         together, with priority-only / assignment-only ablations
                         (results/ga/)
  report_ga.py           figures for the GA report (results/ga/figures/)
  report_similarity.py   Figure 1 of the similarity report (results/similarity/figures/)
  report_paper.py        figures for the IEEE final report (Findings/12_Final_Report/figures/)
  llm_heuristics.py      EoH-style demo: Gemini writes the priority and the machine
                         choice as Python, evolved and then tested on the GA
                         report's instances (results/llm/); key goes in .env
  results/               generated CSVs (overwritten on each run)
```

## Setup

A local environment lives in `.venv/`. To recreate it:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy pandas numba matplotlib
```

`numba` is optional; without it `ga.py` runs much slower.

## Running it

```bash
.venv/Scripts/python run_benchmark.py        # ~25 s
.venv/Scripts/python adversarial_search.py   # several minutes
.venv/Scripts/python ga.py                   # GA experiments; --quick for a smoke test
.venv/Scripts/python llm_heuristics.py --list-models   # models the key in .env can call
.venv/Scripts/python llm_heuristics.py --models gemini-3.8-flash   # ~26 API calls
.venv/Scripts/python llm_heuristics.py --mock          # same pipeline, no API calls
```

`llm_heuristics.py` needs `GEMINI_API_KEY` in `.env` (not shared; listed in
`.gitignore`). On a free-tier key the Pro / Argon / Carbon models have no quota and
stop at once; the Flash models and Gemma work.

`run_benchmark.py` writes four files into `results/`:

| File | Contents |
|---|---|
| `benchmark_by_family.csv` | rule × family, mean % gap to the LP lower bound |
| `benchmark_overall.csv` | rule, overall mean % gap, decoder, group |
| `exact_gaps.csv` | rule vs the *true* optimum on the single-machine family |
| `llm_opportunity.csv` | per family: best single rule, oracle, and the room between them |

`adversarial_search.py` writes the `adversarial_*.csv` files: the two-job
construction, the in-family gap distribution, the worst instance found inside
the generator's value ranges, and the weight-range sweep.

## How the pieces fit

A **heuristic** never builds a schedule. It returns one priority score per job,
and a **decoder** turns that ranking into an admitted set, checking feasibility
itself — so no rule can produce an invalid schedule. Two decoders are provided:

- `greedy_insert` — highest priority first, admit while the set stays feasible.
- `mh_drop` — admit in deadline order, evict the lowest-priority job on overflow.
  With priority `w/p` this is exactly weighted Moore–Hodgson.

Results are reported as **% gap to the LP lower bound**:
`100 * (obj - LB) / (W - LB)`, where `LB` comes from the preemptive relaxation
(Horn's condition). 0% means the bound was matched. On the single-machine family
the exact Lawler–Moore DP is also run, which is how the reports establish that a
large share of that measured gap is slack in the bound rather than error in the
rule.

## Adding a rule

Add an entry to `RULES` in `scheduling/heuristics.py`:

```python
"MyRule": Rule(
    fn=lambda f, c: f["w"] / f["p"],   # higher = keep on time
    decoder="greedy_insert",
    group="week2",
    formula="w_i / p_i",
    idea="one line on what it is doing and what it ignores",
),
```

`f` holds per-job arrays (`p`, `d`, `w`, `wp`, `slack`, `laxity`, `dens`,
`headroom`, `cum_p_before`, …) and `c` instance-level scalars (`m`, `n`, `load`,
`mean_p`, …). Re-run the benchmark; the new rule appears in every table.

## Adding an instance family

Add to `week2_suites()` in `scheduling/suites.py`, using a seed block not already
in use (0–17 and 500–517 are reserved for tuning; 1000–12003 are in use here).
