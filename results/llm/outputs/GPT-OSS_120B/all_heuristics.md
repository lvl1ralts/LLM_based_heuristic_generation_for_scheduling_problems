# Every heuristic GPT-OSS 120B wrote
Called through Groq API (Groq key); OpenAI open-weights model. 18 heuristics written, 16 ran correctly. Training gap = mean % gap to the LP bound on the 12 training instances (lower is better). WSPT scores 4.183%, the hand-written seed 3.503%.
The one marked **tested** has the lowest training gap among the model's own heuristics and is the one run in report 14.

## #0 · generation 0 · seed (hand-written start (WSPT, first fit)) · training gap 3.503%

*Idea:* WSPT priority (weight per unit of processing time); the job goes on the first machine where it fits.

```python
def priority(f, c):
    return f["wp"]


def choose_machine(i, f, c, M):
    return -np.arange(len(M["fits"]))
```

## #1 · generation 0 · i1 (new from scratch) · rejected: ValueError: not allowed: Import

*Idea:* Prioritize jobs with high weight‑per‑time that are tight on their deadline, and assign each job to the feasible machine that keeps the smallest post‑assignment slack while staying lightly loaded.

```python
import numpy as np

def priority(f, c):
    # weight per processing time
    wp = f["wp"]
    # slack (deadline - processing) normalised to [0,1]
    max_slack = np.max(f["slack"])
    slack_norm = f["slack"] / max_slack if max_slack > 0 else np.zeros_like(f["slack"])
    # urgency = p/d  (larger = more urgent)
    urg = f["urgency"]
    # combine: high wp, low slack, high urgency
    # add 1 to avoid zero‑multiplication
    score = wp * (1.0 + (1.0 - slack_norm)) * (1.0 + urg)
    return score

def choose_machine(i, f, c, M):
    # machine is feasible only where M["fits"] is True
    fits = M["fits"]
    # post‑assignment load and slack
    load_after = M["load"] + f["p"][i]
    slack_after = M["slack_after"]          # already -1 where not fit
    # give higher score to smaller slack (tighter) and smaller load
    # shift slack by +1 to keep positivity
    score = np.where(
        fits,
        (1.0 / (load_after + 1e-6)) * (1.0 / (slack_after + 1.0)),
        -np.inf
    )
    return score
```

## #2 · generation 0 · i1 (new from scratch) · training gap 4.939%

*Idea:* Prioritize jobs by weight × laxity per unit processing time, and assign a job to the feasible machine that retains the most slack while staying balanced by load.

```python
def priority(f, c):
    # weight * laxity / processing time
    w = f["w"]
    p = f["p"]
    d = f["d"]
    eps = 1e-9
    laxity = (d - p) / d            # (d‑p)/d, already in f but recomputed for clarity
    score = w * laxity / (p + eps)  # higher is better
    return score


def choose_machine(i, f, c, M):
    # M contains: load, fits, slack_after (‑1 where not fit)
    load = M["load"]
    fits = M["fits"]
    slack = M["slack_after"]

    # start with a score that discards non‑feasible machines
    score = np.full_like(load, -np.inf, dtype=float)

    # indices of feasible machines
    feasible = np.where(fits)[0]
    if feasible.size == 0:
        return score

    # normalize slack and load among feasible machines
    max_slack = np.max(slack[feasible]) + 1e-9
    norm_slack = slack[feasible] / max_slack

    max_load = np.max(load) + 1e-9
    norm_load = load[feasible] / max_load

    # prefer large slack, small load
    score[feasible] = norm_slack - norm_load
    return score
```

## #3 · generation 0 · i1 (new from scratch) · training gap 4.887%

*Idea:* Prioritize jobs by weight‑adjusted slack per unit time and place each job on the feasible machine that preserves the most slack after insertion while keeping its load low.

```python
def priority(f, c):
    # weight * slack per processing time, scaled by normalized headroom
    # avoid division by zero
    eps = 1e-9
    wp = f["w"] * f["slack"] / (f["p"] + eps)
    # incorporate headroom normalization (0 if no headroom info)
    hr = f.get("headroom_norm", np.zeros_like(wp))
    scores = wp * (1 + hr)
    return scores

def choose_machine(i, f, c, M):
    # load after adding job i
    load_after = M["load"] + f["p"][i]
    # slack after adding job i (‑1 where it does not fit)
    slack = M["slack_after"]
    # give non‑feasible machines a very low score
    bad = slack == -1
    # score prefers large slack after and small relative load
    # scale load by max deadline to keep comparable magnitude
    score = slack - load_after / (c["max_d"] + 1e-9)
    score[bad] = -np.inf
    return score
```

## #4 · generation 0 · i1 (new from scratch) · training gap 4.057%

*Idea:* Prioritize jobs that are heavy‑weight per time, urgent, and have little slack, and assign each job to the feasible machine that retains the most slack after insertion while staying lightly loaded.

```python
def priority(f, c):
    """
    Compute a priority score for each job.
    Higher weight‑per‑time, higher urgency, and lower slack increase the score.
    """
    wp = f["wp"]                     # weight / processing time
    urgency = f["urgency"]           # p / d
    slack = f["slack"]               # d - p
    # normalise slack by the largest deadline to keep scale comparable
    slack_norm = slack / float(c["max_d"])
    # combine: heavy & urgent jobs get boost, slack penalises
    score = wp * (1.0 + urgency) / (1.0 + slack_norm)
    return score


def choose_machine(i, f, c, M):
    """
    Score each machine that can still finish job i on time.
    Preference goes to machines that keep the largest slack after insertion
    and have a smaller current load.
    """
    fits = M["fits"]                 # bool array, True where job i fits
    slack_after = M["slack_after"]   # slack remaining on each machine after i
    load = M["load"]                 # current total processing time on each machine

    # give a very low score to infeasible machines
    # score = slack_after / (1 + load)  (more slack, less load → higher score)
    score = np.where(fits, slack_after / (1.0 + load), -np.inf)
    return score
```

## #5 · generation 0 · i1 (new from scratch) · training gap 11.656%

*Idea:* Combine weight‑per‑time with deadline urgency for job ordering, and assign each job to the feasible machine that retains the most slack after insertion while also favouring less loaded machines.

```python
def priority(f, c):
    # Weighted combination of three normalized features:
    #   wp      = weight / processing time
    #   urgency = processing time / deadline
    #   laxity  = (deadline - processing time) / deadline
    # Normalize each feature to [0,1] to keep the scale balanced.
    wp_norm = f["wp"] / (np.max(f["wp"]) + 1e-9)
    urg_norm = f["urgency"] / (np.max(f["urgency"]) + 1e-9)
    lax_norm = f["laxity"] / (np.max(f["laxity"]) + 1e-9)

    # Heuristic coefficients (tuned heuristically)
    alpha, beta, gamma = 0.5, 0.3, 0.2
    scores = alpha * wp_norm + beta * (1 - urg_norm) + gamma * lax_norm
    return scores


def choose_machine(i, f, c, M):
    # M["fits"]  : bool array, True where job i can still be on‑time
    # M["slack_after"] : slack that would remain on each machine after adding i
    # M["load"]        : current total processing time on each machine

    # Prefer machines with larger remaining slack.
    # Slightly penalise heavily loaded machines to improve balance.
    slack_score = np.where(M["fits"], M["slack_after"], -np.inf)
    load_penalty = 0.05 * M["load"]          # small factor, can be tuned
    scores = slack_score - load_penalty
    return scores
```

## #6 · generation 0 · i1 (new from scratch) · training gap 6.144%

*Idea:* Prioritize jobs with high weight‑per‑time that are due early, and assign each job to the feasible machine that leaves the most slack while staying lightly loaded.

```python
def priority(f, c):
    # high weight per processing time, scaled by normalized deadline,
    # penalised by urgency (p/d) so tight deadlines get higher score
    eps = 1e-9
    score = f["wp"] * f["d_norm"] / (f["urgency"] + eps)
    return score


def choose_machine(i, f, c, M):
    # arrays from M
    fits = M["fits"]                     # bool, True where job i fits
    slack = M["slack_after"]             # slack left after placing i (‑1 if not fit)
    load = M["load"]                     # current load on each machine

    # normalise load and slack to make scores comparable
    max_load = np.max(load) if np.max(load) > 0 else 1.0
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0

    load_norm = load / max_load
    slack_norm = slack / max_slack

    # score: favour larger slack and smaller load; infeasible machines get -inf
    score = np.where(fits,
                     slack_norm - 0.5 * load_norm,
                     -np.inf)

    return score
```

## #7 · generation 1 · e1 (new idea, unlike its parents) · training gap 4.636%

*Idea:* Prioritize heavy jobs that belong to tightly‑scheduled periods (low headroom) while favoring short processing times, and place each job on the feasible machine that keeps the resulting load as low as possible, slightly adjusted by the remaining slack.

```python
def priority(f, c):
    """
    Jobs with large weight, short processing time and low headroom (tight schedule)
    receive higher scores.
    """
    w = f["w"]                       # weight
    head = f["headroom_norm"]        # 0 = tight, 1 = loose
    p_norm = f["p_norm"]             # processing time normalised (shorter = smaller)
    eps = 1e-9
    # (1‑head) is larger when headroom is small
    score = w * (1.0 - head) / (p_norm + eps)
    return score


def choose_machine(i, f, c, M):
    """
    Among machines where job i still fits, prefer the one that yields the smallest
    total load after insertion, with a modest bonus for larger slack left.
    """
    fits = M["fits"]                     # bool array, True where job i fits
    load = M["load"]                     # current total processing time per machine
    slack_after = M["slack_after"]       # slack remaining after inserting job i
    p_i = f["p"][i]                      # processing time of job i

    # load after inserting job i
    load_after = load + p_i

    # higher score = better machine
    # -load_after rewards smaller load; 0.1*slack adds a small preference for slack
    score = np.where(fits, -load_after + 0.1 * slack_after, -np.inf)
    return score
```

## #8 · generation 1 · e2 (new idea from what its parents share) · training gap 5.113%

*Idea:* Score jobs by weight efficiency amplified by remaining system headroom and low urgency, and place each job on the feasible machine that maximizes a combined normalized slack‑plus‑minimum‑slack minus load score.

```python
def priority(f, c):
    # weight per time, boosted by headroom (how much capacity is left before its deadline)
    # and penalised by urgency (p/d); higher score = earlier consideration
    eps = 1e-9
    score = f["wp"] * (1.0 + f["headroom_norm"]) * (1.0 - f["urgency"] + eps)
    return score


def choose_machine(i, f, c, M):
    # arrays from M
    fits = M["fits"]                # bool, True where job i fits
    slack = M["slack_after"]        # slack left after placing i (‑1 if not fit)
    load = M["load"]                # current load on each machine
    min_slack = M["min_slack"]      # smallest slack among jobs already on the machine

    # normalise quantities to comparable ranges
    max_slack = np.max(slack[fits]) if np.any(fits) else 1.0
    max_load = np.max(load) if np.max(load) > 0 else 1.0
    max_min_slack = np.max(min_slack) if np.max(min_slack) > 0 else 1.0

    slack_norm = slack / max_slack
    load_norm = load / max_load
    min_slack_norm = min_slack / max_min_slack

    # higher score = more remaining slack and lighter load;
    # infeasible machines receive -inf so they are never chosen
    score = np.where(
        fits,
        slack_norm + 0.5 * min_slack_norm - 0.5 * load_norm,
        -np.inf
    )
    return score
```

## #9 · generation 1 · m1 (changed a parent's logic) · rejected: ValueError: not allowed: Import

*Idea:* Combine weight‑per‑time, deadline urgency, density and laxity for job ordering, and pick the feasible machine that maximizes a slack‑plus‑minimum‑slack score while penalising high load.

```python
import numpy as np
import math

def priority(f, c):
    eps = 1e-9
    # normalised features
    wp_norm   = f["wp"]    / (np.max(f["wp"])    + eps)   # weight / time
    urg_norm  = f["urgency"]/ (np.max(f["urgency"]) + eps)   # p / d  (higher = more urgent)
    dens_norm = f["dens"] / (np.max(f["dens"]) + eps)   # w / (p·d)
    lax_norm  = f["laxity"]/ (np.max(f["laxity"]) + eps)   # (d‑p)/d

    # high wp, low urgency, high density, high laxity are desirable
    scores = (0.4 * wp_norm +
              0.2 * (1.0 - urg_norm) +
              0.2 * dens_norm +
              0.2 * lax_norm)
    return scores


def choose_machine(i, f, c, M):
    eps = 1e-9
    # slack that would remain if the job were placed
    slack = np.where(M["fits"], M["slack_after"], -np.inf)

    # after insertion the smallest slack on the machine (helps keep all jobs on‑time)
    new_min_slack = np.where(M["fits"],
                             np.minimum(M["min_slack"], M["slack_after"]),
                             -np.inf)

    # normalise load by the maximum possible horizon to keep the penalty comparable
    load_penalty = 0.15 * M["load"] / (c["max_d"] + eps)

    # final score: reward slack and the resulting minimum slack, penalise load
    scores = slack + 0.3 * new_min_slack - load_penalty
    return scores
```

## #10 · generation 1 · m2 (changed a parent's constants) · training gap 4.860%

*Idea:* Add headroom‑aware weighted laxity to the priority and balance slack, load, and current minimum slack when picking a machine for finer discrimination.

```python
def priority(f, c):
    # combine weight, laxity and instance headroom (normalized)
    w = f["w"]
    p = f["p"]
    d = f["d"]
    lax = f["laxity"]                     # (d-p)/d
    head_norm = f["headroom_norm"]        # already normalised
    eps = 1e-9

    # higher weight, higher laxity, larger headroom → better;
    # penalise long processing time and high urgency (p/d)
    urgency = f["urgency"]                # p/d
    score = (w * lax * (1.0 + head_norm)) / ((p + eps) * (1.0 + urgency))
    return score


def choose_machine(i, f, c, M):
    # machine‑wise scores: keep slack, keep load balanced, preserve min slack
    load   = M["load"]          # total processing time on each machine
    fits   = M["fits"]          # bool array, True where job i fits
    slack  = M["slack_after"]   # slack remaining after adding i (‑1 if not fit)
    min_sl = M["min_slack"]     # current smallest slack on each machine

    # initialise with -inf so non‑feasible machines are never chosen
    score = np.full_like(load, -np.inf, dtype=float)

    feasible = np.where(fits)[0]
    if feasible.size == 0:
        return score

    # normalise the three components among feasible machines
    max_slack = np.max(slack[feasible]) + 1e-9
    norm_slack = slack[feasible] / max_slack

    max_load = np.max(load) + 1e-9
    norm_load = load[feasible] / max_load

    max_min_slack = np.max(min_sl[feasible]) + 1e-9
    norm_min_slack = min_sl[feasible] / max_min_slack

    # weighted combination: favour slack, penalise load, also keep min slack high
    alpha, beta, gamma = 0.55, 0.30, 0.15
    score[feasible] = alpha * norm_slack - beta * norm_load + gamma * norm_min_slack
    return score
```

## #11 · generation 2 · e1 (new idea, unlike its parents) · training gap 5.181%

*Idea:* Prioritize jobs with a high weight‑to‑slack ratio and early deadlines, and place each job on the feasible machine that preserves the largest minimum slack while keeping load low.

```python
def priority(f, c):
    """
    Jobs that are heavy, have little slack and early deadlines get higher scores.
    """
    w     = f["w"]
    slack = f["slack"]          # d - p
    d     = f["d"]
    head  = f["headroom_norm"]  # 0 = tight, 1 = loose
    eps   = 1e-9

    # weight per unit slack, penalise large deadlines and loose headroom
    score = (w / (slack + eps)) * (1.0 - head) * (1.0 / (d + eps))
    return score


def choose_machine(i, f, c, M):
    """
    Among machines where job i still fits, favour the one that keeps the
    smallest slack on that machine as large as possible and that does not
    become overloaded.
    """
    fits          = M["fits"]          # bool array
    load          = M["load"]          # current total processing time
    slack_after   = M["slack_after"]   # slack left after adding i (‑1 if not fit)
    min_slack_cur = M["min_slack"]     # current minimum slack on each machine

    # new minimum slack after insertion (only where the job fits)
    new_min_slack = np.where(fits, np.minimum(min_slack_cur, slack_after), -np.inf)

    # load after inserting the job
    p_i       = f["p"][i]
    load_after = load + p_i

    # initialise scores with -inf (non‑feasible machines stay at -inf)
    score = np.full_like(load, -np.inf, dtype=float)

    feasible = np.where(fits)[0]
    if feasible.size == 0:
        return score

    # normalise the two components among feasible machines
    max_min = np.max(new_min_slack[feasible]) + 1e-9
    norm_min = new_min_slack[feasible] / max_min

    max_load = np.max(load_after[feasible]) + 1e-9
    norm_load = load_after[feasible] / max_load

    # combine: keep min slack high (alpha) and load low (beta)
    alpha, beta = 0.7, 0.3
    score[feasible] = alpha * norm_min - beta * norm_load
    return score
```

## #12 · generation 2 · e2 (new idea from what its parents share) · training gap 5.785%

*Idea:* Prioritize jobs by a weight‑per‑time measure tempered by how tight the schedule is (headroom) and deadline urgency, and assign each job to the feasible machine that maximizes the ratio of remaining slack to resulting load.

```python
def priority(f, c):
    """
    High weight‑per‑processing time, early/urgent jobs in a tight schedule get higher scores.
    """
    wp      = f["wp"]                     # w / p
    head    = f["headroom_norm"]          # 0 = tight, 1 = loose
    urg     = f["urgency"]                # p / d
    d       = f["d"]
    eps = 1e-9

    # combine: reward large wp, penalise loose headroom and low urgency (large p/d)
    score = wp * (1.0 - head) * (1.0 / (urg + eps)) * (1.0 / (d + eps))
    return score


def choose_machine(i, f, c, M):
    """
    Among machines where job i still fits, prefer the one that keeps the
    slack‑to‑load ratio largest after insertion.
    """
    fits          = M["fits"]           # bool array
    slack_after   = M["slack_after"]    # slack left after adding i, -1 if not fit
    load          = M["load"]           # current total processing time
    p_i           = f["p"][i]

    # load after inserting the job (invalid where not fit will be ignored later)
    load_after = load + p_i

    # initialise with -inf for infeasible machines
    score = np.full_like(load, -np.inf, dtype=float)

    feasible = np.where(fits)[0]
    if feasible.size == 0:
        return score

    # normalize slack and load among feasible machines
    max_slack = np.max(slack_after[feasible]) + 1e-9
    max_load  = np.max(load_after[feasible]) + 1e-9

    norm_slack = slack_after[feasible] / max_slack
    norm_load  = load_after[feasible] / max_load

    # larger slack and smaller load are better: use ratio
    score[feasible] = norm_slack / (norm_load + 1e-9)
    return score
```

## #13 · generation 2 · m1 (changed a parent's logic) · training gap 4.149%

*Idea:* Assign a job to the feasible machine that leaves the most slack after scheduling it, breaking ties by preferring the less‑loaded machine.

```python
def priority(f, c):
    # keep the effective WSPT rule
    return f["wp"]


def choose_machine(i, f, c, M):
    # slack_after is the smallest remaining slack on the machine after adding job i
    # higher slack is better; if equal, prefer the machine with lower current load
    scores = M["slack_after"].astype(float)
    scores -= 0.001 * M["load"]          # small penalty for higher load
    return scores
```

## #14 · generation 2 · m2 (changed a parent's constants) · training gap 4.710%

*Idea:* Add normalized weight, inverse slack, early deadline and urgency components with tuned coefficients, and score machines by a weighted sum of normalized new minimum slack, remaining slack after insertion and inverse load.

```python
def priority(f, c):
    """
    Composite score: heavy jobs, little slack, early deadlines and high urgency
    are favoured; tight global headroom also raises priority.
    """
    eps = 1e-9
    w      = f["w"]
    slack  = f["slack"]          # d - p
    d      = f["d"]
    urg    = f["urgency"]        # p / d
    head   = f["headroom_norm"]  # 0 = tight, 1 = loose

    # normalised components
    w_n      = w / (np.max(w) + eps)
    inv_sl   = 1.0 / (slack + eps)
    inv_sl_n = inv_sl / (np.max(inv_sl) + eps)
    d_n      = 1.0 / (d + eps)
    d_n      = d_n / (np.max(d_n) + eps)
    urg_n    = urg / (np.max(urg) + eps)
    tight_n  = 1.0 - head        # high when headroom is tight

    # weighted sum (tuned)
    score = (0.35 * w_n +
             0.30 * inv_sl_n +
             0.20 * d_n +
             0.10 * urg_n +
             0.05 * tight_n)

    return score


def choose_machine(i, f, c, M):
    """
    For a job that fits on several machines, prefer the machine that
    (1) keeps the smallest slack on that machine large,
    (2) leaves the most slack after inserting the job,
    (3) stays lightly loaded.
    """
    eps = 1e-9
    fits          = M["fits"]          # bool array
    load          = M["load"]          # current total processing time
    min_slack_cur = M["min_slack"]     # current minimum slack per machine
    slack_after   = M["slack_after"]   # slack left after adding i (‑1 if not fit)

    # new minimum slack after insertion (only where the job fits)
    new_min_slack = np.where(fits,
                             np.minimum(min_slack_cur, slack_after),
                             -np.inf)

    # normalise the three criteria among feasible machines
    feasible = np.where(fits)[0]
    if feasible.size == 0:
        return np.full_like(load, -np.inf, dtype=float)

    # 1) normalized new minimum slack (higher better)
    max_min = np.max(new_min_slack[feasible]) + eps
    norm_min = new_min_slack[feasible] / max_min

    # 2) normalized remaining slack after insertion (higher better)
    max_rem = np.max(slack_after[feasible]) + eps
    norm_rem = slack_after[feasible] / max_rem

    # 3) normalized load after insertion (lower better)
    load_after = load + f["p"][i]
    max_load = np.max(load_after[feasible]) + eps
    norm_load = load_after[feasible] / max_load

    # weighted combination
    a, b, c_coeff = 0.45, 0.35, 0.20   # a: min slack, b: remaining slack, c: load
    score_feas = a * norm_min + b * norm_rem - c_coeff * norm_load

    # assemble full score vector
    score = np.full_like(load, -np.inf, dtype=float)
    score[feasible] = score_feas
    return score
```

## #15 · generation 3 · e1 (new idea, unlike its parents) · training gap 6.030%

*Idea:* Prioritize jobs that are dense in weight per unit of processing‑time‑deadline product and have early deadlines; assign each job to the feasible machine that preserves the most slack after insertion while keeping load low.

```python
def priority(f, c):
    """
    Score = density (w/(p*d)) boosted for early deadlines.
    """
    dens = f["dens"]                     # weight per (p*d)
    d_rank = f["d_rank"]                 # 0 = earliest deadline, 1 = latest
    return dens * (1.0 - d_rank)


def choose_machine(i, f, c, M):
    """
    Among machines where job i fits, prefer the one that leaves the largest
    slack after insertion, with a small penalty for higher total load.
    """
    fits = M["fits"]                     # bool array
    slack_after = M["slack_after"]       # slack that would remain after adding i
    load = M["load"]                     # current load on each machine
    p_i = f["p"][i]                      # processing time of job i

    # total load after insertion
    load_after = load + p_i

    # higher score = better machine
    # keep large slack, slightly discourage heavy load
    score = np.where(fits,
                     slack_after - 0.01 * load_after,
                     -np.inf)
    return score
```

## #16 · generation 3 · e2 (new idea from what its parents share) · training gap 5.166%

*Idea:* Use a density‑based priority that also rewards laxity and instance headroom, and pick the machine that maximises a weighted blend of remaining slack, low current load and high current minimum slack.

```python
def priority(f, c):
    # density = w/(p*d); boost by laxity and normalised headroom
    dens   = f["dens"]                     # w/(p*d)
    lax    = f["laxity"]                   # (d-p)/d
    head   = f["headroom_norm"]            # already normalised
    eps = 1e-9
    # higher weight, larger laxity and more headroom → better score
    return dens * (1.0 + lax) * (1.0 + head + eps)


def choose_machine(i, f, c, M):
    # start with impossible score for non‑feasible machines
    score = np.full_like(M["load"], -np.inf, dtype=float)

    feasible = np.where(M["fits"])[0]
    if feasible.size == 0:
        return score

    # extract relevant vectors for feasible machines
    slack    = M["slack_after"][feasible]      # remaining slack after adding i
    load     = M["load"][feasible]             # current load
    min_sl   = M["min_slack"][feasible]        # current minimum slack

    # normalise each component among feasible machines
    max_slack = np.max(slack) + 1e-9
    norm_slack = slack / max_slack

    # load: prefer smaller values → invert and normalise
    min_load = np.min(load)
    max_load = np.max(load) + 1e-9
    norm_load = 1.0 - (load - min_load) / (max_load - min_load + 1e-9)

    max_min_sl = np.max(min_sl) + 1e-9
    norm_min_sl = min_sl / max_min_sl

    # combine with new weights
    alpha, beta, gamma = 0.45, 0.35, 0.20
    score[feasible] = alpha * norm_slack + beta * norm_load + gamma * norm_min_sl
    return score
```

## #17 · generation 3 · m1 (changed a parent's logic) · training gap 3.921% — **tested**

*Idea:* Improve priority by combining weight, weight‑per‑processing, tight headroom and urgency, and balance machines by also considering current load, remaining slack and number of jobs on each machine.

```python
def priority(f, c):
    """
    High weight, high weight‑per‑processing, tight headroom and high urgency
    increase the score; short processing time further helps.
    """
    eps = 1e-9
    w      = f["w"]
    wp     = f["wp"]                 # w / p
    head   = f["headroom_norm"]      # 0 = tight, 1 = loose
    urg    = f["urgency"]            # p / d
    p_norm = f["p_norm"]             # normalised processing time

    # (1‑head) favours tight periods, urg favours early‑deadline jobs
    score = w * wp * (1.0 - head) * (1.0 + urg) / (p_norm + eps)
    return score


def choose_machine(i, f, c, M):
    """
    Among feasible machines, prefer the one that keeps total load low,
    leaves more slack, and balances the number of assigned jobs.
    """
    fits   = M["fits"]                # bool array
    load   = M["load"]                # current load per machine
    slack  = M["slack_after"]         # slack after inserting job i
    count  = M["count"]               # number of jobs already on the machine
    p_i    = f["p"][i]

    load_after = load + p_i

    # higher score = better machine
    # -load_after  : minimise load
    # +0.2*slack   : reward remaining slack
    # -0.5*count   : spread jobs evenly
    score = np.where(fits,
                     -load_after + 0.2 * slack - 0.5 * count,
                     -np.inf)
    return score
```

## #18 · generation 3 · m2 (changed a parent's constants) · training gap 4.278%

*Idea:* Combine normalized weight‑per‑time, urgency, headroom and laxity while penalising slack, and select the feasible machine that maximises slack‑after weighted by headroom and inversely by current load.

```python
def priority(f, c):
    """
    Score jobs: high weight‑per‑time, urgency, headroom and laxity are good;
    large slack is bad.
    """
    wp_norm = f["wp"] / np.max(f["wp"])
    urgency = f["urgency"]
    headroom = f["headroom_norm"]
    laxity = f["laxity"]
    slack_norm = f["slack"] / float(c["max_d"])          # 0..≈1
    # combine multiplicatively, penalise slack
    score = wp_norm * (1.0 + urgency) * (1.0 + headroom) * laxity / (1.0 + slack_norm)
    return score


def choose_machine(i, f, c, M):
    """
    Score each feasible machine for job i.
    Prefer machines that keep more slack after insertion and are lightly loaded,
    boosted by the job's headroom factor.
    """
    fits = M["fits"]                         # bool array
    slack_after = M["slack_after"]           # remaining slack after adding i
    load = M["load"]                         # current total processing time
    # normalise slack to the same scale used in priority
    slack_norm = slack_after / float(c["max_d"])
    head = f["headroom_norm"][i]             # per‑job headroom factor
    # high slack_norm, high headroom, low load → high score
    score = np.where(fits,
                     slack_norm * (1.0 + head) / (1.0 + load),
                     -np.inf)
    return score
```
