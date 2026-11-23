# BTP demo: an LLM writes a scheduling rule, live

One LLM request, one instance, about 10 seconds end to end.

```
cd D:\Lelouch\BTP\btp_demo
python demo.py            live call to GPT-OSS 120B on Groq (answers in 2 to 3 s)
python demo.py --saved    no internet: uses Gemma's rule from the report
```

First time on a computer: `python -m pip install -r requirements.txt` (numpy, scipy, pandas).
On this laptop they are already installed. `run_demo.bat` does the same thing with the
project's own Python (`..\Code\.venv`).

Other options: `--model groq:qwen/qwen3.8-27b` or `--model gemini-3.8-flash` (Gemini can take a
minute or more, or fail with "high demand", so Groq is the default for a live demo), and
`--limit 60` to wait at most 60 s for the LLM. The script still uses the code in `../Code`
and the API keys in `../Code/.env`, so keep the two folders side by side.

If a reply is unusable (for example the model adds an `import`, which the safety check
refuses), the script says so and asks again, up to three times, before using Gemma's rule.

## What it does

1. **Instance.** `instance.json`: the 6-job, 2-machine MetaOpt instance from report 14 and
   the final report (Fig. 6). Total weight 37; the optimum, found with the HiGHS integer
   program, loses 1.
2. **WSPT.** The best classical rule, with the usual earliest-free-machine placement. It
   loses 11: J5 and the valuable long job J6 are late.
3. **The LLM request.** The same request that produced Gemma's rule in the report: the task
   description (the problem, the two Python functions to write, the inputs they can read),
   one earlier heuristic (WSPT + "machine with the most slack left"), and *"Change its logic
   to make it better. Keep what works."* The full prompt is written to `output/prompt.txt`.
4. **The heuristic.** The model's one-line idea and its two functions, `priority` and
   `choose_machine`. Before running, the code is checked (no imports, no file access) and
   then run in a separate process.
5. **Run.** The schedule it produces on the instance, and its mean gap on the 12 training
   instances.
6. **Comparison.** Gemma's rule from our full run (WSPT + busiest machine that fits) on the
   same instance: it loses 1, the optimum.

`output/` keeps the prompt, the raw reply, the heuristic as a `.py` file and the numbers, one
set per run (the files from test runs on 15 Nov can be deleted).

If the call fails (no internet, quota), the script says so and uses Gemma's rule instead, so
the demo never stops halfway. API keys are read from `../Code/.env` and never printed.

## What to expect from the live model

GPT-OSS answers fast but usually keeps the parent's "spread jobs out" placement, so on this
instance it tends to lose 11, like WSPT. That is fine and worth saying: one request is one
candidate, and the loop's job is to score many candidates and keep the good ones. Gemma's
rule, the one that packs the busiest machine, is what that selection found in our full run,
and step 6 shows it next to the live one.
