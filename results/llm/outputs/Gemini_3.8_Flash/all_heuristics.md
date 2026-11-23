# Every heuristic Gemini 3.8 Flash wrote
Called through Google Gemini API (Gemini key). 6 heuristics written, 6 ran correctly. Training gap = mean % gap to the LP bound on the 12 training instances (lower is better). WSPT scores 4.183%, the hand-written seed 3.503%.
The one marked **tested** has the lowest training gap among the model's own heuristics and is the one run in report 14.

## #0 · generation 0 · seed (hand-written start (WSPT, first fit)) · training gap 3.503%

*Idea:* WSPT priority (weight per unit of processing time); the job goes on the first machine where it fits.

```python
def priority(f, c):
    return f["wp"]


def choose_machine(i, f, c, M):
    return -np.arange(len(M["fits"]))
```

## #1 · generation 0 · i1 (new from scratch) · training gap 3.935%

*Idea:* Prioritize high value-density jobs scaled by urgency and early deadlines, and assign jobs to busy machines where they finish early while preserving slack.

```python
def priority(f, c):
    return f["wp"] * (1.0 + np.sqrt(f["urgency"])) * (2.0 - f["d_rank"])

def choose_machine(i, f, c, M):
    return 1.2 * M["load"] - M["finish_after"] + 0.1 * M["slack_after"]
```

## #2 · generation 0 · i1 (new from scratch) · training gap 4.206%

*Idea:* Prioritize jobs by weight-to-processing-time discounted by relative deadline, and assign to the feasible machine that maximizes remaining slack while minimizing the job's completion time.

```python
def priority(f, c):
    return (f["wp"] * (1.0 + f["urgency"])) / (f["d_norm"] + 0.5)

def choose_machine(i, f, c, M):
    return M["slack_after"] - M["finish_after"]
```

## #3 · generation 0 · i1 (new from scratch) · training gap 4.225%

*Idea:* Prioritize jobs by value density smoothed by deadline urgency, and assign each job to the feasible machine that maximizes remaining slack while minimizing completion time.

```python
def priority(f, c):
    return f["wp"] / (np.sqrt(f["d_norm"] + 0.05) * (1.0 + np.maximum(0.0, f["headroom_norm"])))

def choose_machine(i, f, c, M):
    return M["slack_after"] - 0.5 * M["finish_after"]
```

## #4 · generation 1 · e1 (new idea, unlike its parents) · training gap 3.196% — **tested**

*Idea:* Prioritize jobs primarily by weight density with a gentle boost for earlier deadline rank, and place each job on the feasible machine with the highest current load to pack utilized machines and preserve empty ones.

```python
def priority(f, c):
    return f["wp"] * (1.0 + 0.1 * (1.0 - f["d_rank"]))


def choose_machine(i, f, c, M):
    return M["load"] - 1e-4 * np.arange(len(M["load"]))
```

## #5 · generation 1 · e2 (new idea from what its parents share) · training gap 3.281%

*Idea:* Prioritize jobs by weight density with mild bonuses for earlier deadline rank and smaller processing time, and assign each job to the feasible machine that maximizes load while preserving post-insertion slack.

```python
def priority(f, c):
    return (
        f["wp"]
        * (1.0 + 0.12 * (1.0 - f["d_rank"]))
        * (1.0 + 0.04 * (1.0 - f["p_norm"]))
    )


def choose_machine(i, f, c, M):
    return M["load"] + 1e-4 * M["slack_after"] - 1e-6 * np.arange(c["m"])
```

## #6 · generation 1 · m1 (changed a parent's logic) · training gap 4.751%

*Idea:* Boost priority for jobs in oversubscribed deadline intervals by allowing negative headroom, and penalize both completion time and total machine load when assigning machines.

```python
def priority(f, c):
    return f["wp"] / (np.sqrt(f["d_norm"] + 0.05) * (1.0 + np.clip(f["headroom_norm"], -0.6, 2.0)))

def choose_machine(i, f, c, M):
    return M["slack_after"] - 0.4 * M["finish_after"] - 0.2 * M["load"]
```

## #7 · generation 1 · m2 (changed a parent's constants) · API call failed, nothing written: RuntimeError: quota used up for gemini-3.8-flash (resets in 21 h); a free-tier k
