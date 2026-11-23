# Every heuristic Qwen 3.8 27B wrote
Called through Groq API (Groq key); Alibaba open-weights model. 18 heuristics written, 9 ran correctly. Training gap = mean % gap to the LP bound on the 12 training instances (lower is better). WSPT scores 4.183%, the hand-written seed 3.503%.
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

*Idea:* Assign jobs in decreasing order of a combined metric that favors high weight, low processing time, and tight deadlines, then place each job on the feasible machine that leaves the most future flexibility (highest minimum slack after placement) while balancing current load.

```python
import numpy as np

def priority(f, c):
    # Combine urgency (tight deadline relative to work), benefit (weight), and efficiency
    # Use normalized components to balance scales
    w_norm = f["w_norm"]
    d_norm = f["d_norm"]
    p_norm = f["p_norm"]
    
    # High weight, low deadline (urgent), low processing time (efficient)
    # Invert d_norm and p_norm to make high score = urgent/short
    score = w_norm * (2.0 - d_norm) * (1.0 + p_norm)
    return score

def choose_machine(i, f, c, M):
    # Among feasible machines (fits == True), choose the one with:
    # 1. Highest slack_after (most buffer for future jobs)
    # 2. Tie-break: lowest current load (balance)
    fits = M["fits"]
    slack_after = M["slack_after"].copy()
    load = M["load"].copy()
    
    # Set infeasible machines to very low score
    # We want to maximize slack_after, but penalize infeasible
    # Since slack_after is -1 where doesn't fit, we can use it directly.
    # However, to prioritize slack over load in a simple formula:
    # Score = slack_after - epsilon * (load / total_p)
    # But we need to ensure infeasible (slack=-1) is less than feasible.
    # Let's just pick max slack_after among fits. If tie, min load.
    
    # Create score: Primary: slack_after, Secondary: -load
    # To handle tie-breaking in one numpy operation without sorting complexities:
    # We can rank by slack, then break ties by load.
    # Simple approach: filter to fits, then find indices.
    
    feasible_indices = np.where(fits)[0]
    if len(feasible_indices) == 0:
        # Should not happen per problem description ("called only when two or more machines fit")
        return np.array([0])
        
    # Get slack and load for feasible
    feas_slack = slack_after[feasible_indices]
    feas_load = load[feasible_indices]
    
    # Primary key: max slack
    max_slack = np.max(feas_slack)
    # Secondary key: min load among those with max slack
    target_slack = feas_slack[feas_slack == max_slack]
    if len(target_slack) > 0:
        min_load = np.min(feas_load[feas_slack == max_slack])
        # Find first index that satisfies both
        # We just return an array of scores where the maximum score corresponds to the best machine
        # Or we can identify the machine and create a score array.
        # The problem says: "return numpy array of m scores... job goes on... highest score"
        # So we just need to assign a high score to the chosen machine.
        
        # Construct score. High score for best machine.
        scores = np.full(c["m"], -np.inf)
        # Get indices of machines with max_slack
        best_indices = feasible_indices[feas_slack == max_slack]
        # Among those, get the one with min load
        loads_of_best = feas_load[feas_slack == max_slack]
        best_machine_rel_idx = np.argmin(loads_of_best)
        best_machine_abs_idx = best_indices[best_machine_rel_idx]
        
        scores[best_machine_abs_idx] = 1.0
        # Optional: give slightly lower scores to other feasible to make "highest" unambiguous if needed,
        # but 1.0 vs -inf is enough.
        return scores
    
    # Fallback (shouldn't reach here)
    return np.array([0])
```

## #2 · generation 0 · i1 (new from scratch) · rejected: ValueError: not allowed: Import

*Idea:* Schedule jobs by decreasing density (w/(p*d)) and priority slack, preferring machines with the least negative slack_after among feasible ones to balance deadlines and load.

```python
import numpy as np
import math

def priority(f, c):
    # Combine density and slack: prioritize high density and high slack (less urgent slack? No, high slack means more buffer, but we want high density. Let's use density * slack or similar. 
    # Actually, standard heuristics often use Earliest Due Date or Modified Due Date. 
    # Let's try: weight * urgency * slack. Or w * (1 - urgency). 
    # A common good heuristic is Weighted Shortest Processing Time or Highest Density.
    # Let's use: w * slack. Higher weight and more slack (more time to finish) is good? 
    # Wait, if slack is high, it's less urgent. If high, we should do it later? 
    # No, the function returns scores where jobs are tried in DECREASING score. 
    # So we want jobs that are "best" first. 
    # Best usually means: high weight, tight deadline (low slack? or high urgency?). 
    # If we pick a job with low slack first, it's safe. 
    # Let's try: score = w * slack. If a job has large slack, it has lots of time. 
    # But if we delay it, it might become tight. 
    # Actually, standard "Highest Density" (w/(pd)) is good. 
    # Let's mix density and weight. 
    # Score = f["w"] * f["slack"] + f["dens"]
    # Let's stick to a simple, robust metric: Weighted Slack.
    return f["w"] * f["slack"] + f["dens"] * f["w"]

def choose_machine(i, f, c, M):
    # We want to choose the machine where the job fits and results in the "best" state.
    # We prefer machines where the job leaves the most slack for other jobs? 
    # Or where the minimum slack after adding is highest? 
    # "slack_after" is the smallest d - finish_time on that machine after adding job i.
    # We want this value to be as high as possible (positive slack is good). 
    # If no machine fits, slack_after is -1. 
    # The prompt says: "Job i is offered only to the machines where it still fits". 
    # And "If no machine fits, the job is late." 
    # The function returns scores for m machines. 
    # We should give a high score to feasible machines. 
    # We want to maximize the slack_after among feasible machines? 
    # Or minimize the load? 
    # Choosing the machine with the highest min slack after insertion seems to preserve future flexibility. 
    # If a machine is very loaded (low slack), adding a job might make it negative (late). 
    # But the input M["fits"] already handles feasibility. 
    # So among M["fits"] == True, we want the one with the highest M["slack_after"]. 
    # If a machine doesn't fit, give it a very low score (e.g., -inf or -1e10). 
    
    fits = M["fits"]
    slack_after = M["slack_after"]
    
    # Initialize scores to a very low value
    scores = np.full(c["m"], -1e10)
    
    # For feasible machines, set score to slack_after
    # We want to pick the machine with the highest slack_after
    scores[fits] = slack_after[fits]
    
    return scores
```

## #3 · generation 0 · i1 (new from scratch) · training gap 6.784%

*Idea:* Maximize job weight per unit processing time (density) while minimizing required processing time, and assign to the machine with the most remaining minimum slack to preserve future feasibility.

```python
def priority(f, c):
    p = f["p"]
    w = f["w"]
    wp = f["wp"]
    # High density (w/p) is good, small processing time is good
    # Normalize p to avoid scale issues, use p_norm if available or compute
    p_max = np.max(p) if len(p) > 0 else 1.0
    p_norm = p / p_max
    
    score = wp - 0.2 * p_norm
    return score

def choose_machine(i, f, c, M):
    slack_after = M["slack_after"]
    fits = M["fits"]
    
    # Only consider feasible machines
    if not np.any(fits):
        # Should not be called if no machine fits, but handle gracefully
        return np.zeros(len(slack_after))
        
    # Mask out infeasible machines with very low score
    # We want high slack_after (more free time before deadlines)
    # We might also prefer machines with less load to balance, but slack is key for feasibility
    # Let's prioritize higher slack_after.
    # To break ties or add slight balance, we could look at load, but slack_after captures the constraint.
    
    # Create scores: infeasible get 0, feasible get slack_after
    scores = np.where(fits, slack_after, 0.0)
    
    return scores
```

## #4 · generation 0 · i1 (new from scratch) · rejected: ValueError: not allowed: Import

*Idea:* Maximize weight by prioritizing jobs with high urgency and density, and placing each job on the machine that leaves the most slack after placement to maintain future flexibility.

```python
import numpy as np

def priority(f, c):
    # Combine urgency (p/d) and density (w/(p*d)) with weight to favor
    # jobs that are time-critical and have high value per time unit.
    # Normalize components to be in [0, 1] range for stable combination.
    urgency = f["urgency"]
    dens = f["dens"]
    w_norm = f["w_norm"]
    
    # We want a heuristic that balances time criticality (urgency) and value density.
    # A simple linear combination where urgency drives the scheduling feasibility
    # and density drives the profit aspect.
    # Let's use a convex combination: 0.5 * urgency + 0.5 * dens, scaled by weight norm?
    # Actually, simple EDD (Earliest Deadline First) is a strong baseline for completeness.
    # However, we want to maximize weight. 
    # Let's tryWeighted Shortest Processing Time or similar, but deadlines matter.
    # A common approach is to sort by deadline (EDD) initially, but here we need a score.
    # Let's use a score that heavily penalizes large deadlines (high urgency) 
    # and rewards high density.
    
    # Normalize urgency and dens to [0,1] just to be safe, though they are already normalized-ish.
    # f["urgency"] is p/d. Max urgency is 1 (if p=d). Min is small.
    # f["dens"] is w/(p*d).
    
    # Let's create a score: high urgency is good. High density is good.
    # Score = urgency + dens
    # But we must ensure the size matches n.
    
    # To make it more robust, let's look at "headroom" or "slack".
    # Jobs with tight slack should probably be processed first to avoid being late.
    # But the prompt says "Jobs are tried in decreasing score".
    # If a job has very low slack, it's risky to delay it. So it should have high priority.
    # slack = d - p. High slack means it can wait. Low slack means it can't.
    # So priority should be inversely proportional to slack?
    # Or we can just use 1/slack? But slack can be 0.
    # Laxity = (d-p)/d. High laxity means lots of time to spare. Low laxity means critical.
    # So high priority for low laxity. 
    # Score = 1 / (laxity + epsilon)?
    
    # Let's keep it simple and interpretable:
    # Prioritize jobs with high urgency (p/d) and high density (w/(p*d)).
    # This balances the "tightness" of the job regarding its own duration vs deadline
    # and the economic efficiency.
    
    score = f["urgency"] + f["dens"]
    return score

def choose_machine(i, f, c, M):
    # We are given M["fits"] (bool), M["slack_after"] (smallest slack if job added, -1 if not fit),
    # and M["finish_after"].
    # We should pick the machine where the job fits and results in the best "slack_after".
    # Maximizing slack_after on the machine ensures that the tightest constraint on that machine
    # is as loose as possible after adding this job. This is a form of "most flexible" placement.
    
    fits = M["fits"]
    slack_after = M["slack_after"]
    
    # Where it doesn't fit, slack_after is -1.
    # We want the maximum slack_after among those that fit.
    # If no machine fits, the problem statement says "If no machine fits, the job is late."
    # But the function is called ONLY when two or more machines fit.
    # So we can assume at least two +inf values effectively.
    
    # To select the machine with the highest slack_after (meaning we leave the most room for future jobs),
    # we assign a score based on slack_after.
    # Scores should be higher for better machines.
    
    # If it fits, score = slack_after. If not, score = -inf (or a very low number).
    # Since only fitting machines are considered for "highest score", and the problem says
    # "job is offered only to the machines where it still fits", the `fits` array
    # in M might indicate which are offered. However, the prompt says "choose_machine... is called only when
    # two or more machines fit". This implies the caller filters, or we handle the -1s.
    # Let's assume we should return scores for all m machines, and the caller picks the max.
    # The prompt says: "The job goes on the feasible machine with the highest score."
    # So we need to return an array of m scores.
    
    # We create a score array.
    scores = np.full(M["load"].shape, -np.inf)
    
    # For machines that fit, assign the slack_after as the score.
    # Higher slack_after is better because it means the machine's tightest job in the sequence
    # still has plenty of time before its deadline.
    fitting_idx = np.where(fits)[0]
    if len(fitting_idx) > 0:
        scores[fitting_idx] = slack_after[fitting_idx]
        
    return scores
```

## #5 · generation 0 · i1 (new from scratch) · training gap 6.893%

*Idea:* Prioritize jobs by high density (weight/processing time) and low deadline rank to maximize weight on tight deadlines, placing them on the machine with the highest residual slack after placement to maintain schedule feasibility.

```python
def priority(f, c):
    # High density and early deadline (low rank) are preferred
    # d_rank is 0..1 where 0 is earliest
    score = f["dens"] * (1.0 - f["d_rank"])
    return score

def choose_machine(i, f, c, M):
    # Only consider machines where the job fits
    fits_mask = M["fits"]
    # Among fitting machines, prefer those with higher residual slack
    # slack_after is the smallest slack on machine after adding job i
    # We want to maximize slack_after to leave more room for future jobs
    # If slack_after is -1 (should not happen for fits=True), treat as -inf
    scores = np.where(fits_mask, M["slack_after"], -np.inf)
    return scores
```

## #6 · generation 0 · i1 (new from scratch) · training gap 4.147%

*Idea:* Assign jobs by decreasing density of weight per unit time, placing each on the machine that maximizes remaining slack after placement to preserve feasibility for tighter future jobs.

```python
def priority(f, c):
    p = f["p"]
    w = f["w"]
    d = f["d"]
    # Fall back to weight/density if processing time is zero to avoid division by zero
    safe_p = np.where(p == 0, 1e-9, p)
    return w / safe_p

def choose_machine(i, f, c, M):
    slack_after = M["slack_after"]
    # Only consider machines where the job fits (slack_after >= 0)
    # M["fits"] is boolean, but slack_after >= 0 implies fits based on problem description
    # Higher slack is better
    scores = slack_after.copy().astype(float)
    # Set infeasible machines to -infinity so they are never chosen
    fits = M["fits"]
    scores[~fits] = -np.inf
    return scores
```

## #7 · generation 0 · i1 (new from scratch) · rejected: ValueError: not allowed: Import

*Idea:* Prioritize jobs by the ratio of weight to deadline to capture high-value urgent work, and assign each job to the feasible machine with the least remaining load to balance the schedule and minimize future contention.

```python
import numpy as np
import math

def priority(f, c):
    # Prioritize jobs with high weight relative to their deadline.
    # Higher weight/d means more urgent or valuable per unit of time constraint.
    return f["w"] / f["d"]

def choose_machine(i, f, c, M):
    # M["fits"] is a bool array. We only consider machines where fits is True.
    # Among feasible machines, we want the one that minimizes the remaining load
    # or minimizes the maximum expected finish time to balance the load.
    # M["load"] contains the current load.
    # However, M["slack_after"] indicates the smallest slack if job i is added.
    # Maximizing slack_after is equivalent to minimizing the tightness.
    # Alternatively, minimizing M["load"] balances the total processing time.
    # Let's use load balancing: minimize the current load to spread work evenly.
    
    # Initialize scores with a very negative number so infeasible machines are ignored
    scores = np.full(M["load"].shape, -np.inf)
    feasible_mask = M["fits"]
    
    # Among feasible machines, choose the one with the lowest load (most capacity left)
    # This is a standard LPT (Longest Processing Time) like heuristic in terms of distribution,
    # though here we are assigning sequentially. Actually, minimizing load is good for balance.
    # Another option: maximize slack_after (most feasibility remaining).
    # Let's go with maximizing slack_after, as it prioritizes keeping options open for the tightest jobs.
    
    # If we minimize load, we might overload slightly later. If we maximize slack_after,
    # we place the job on the machine where it fits most comfortably (largest slack).
    
    # Let's look at the definition: "The job goes on the feasible machine with the highest score."
    # So higher score is better.
    
    # Strategy: Choose the machine that maximizes the slack_after.
    # slack_after is the smallest slack on the machine if job i is added.
    # Larger slack_after means the machine has more "breathing room" for future jobs.
    
    scores[feasible_mask] = M["slack_after"][feasible_mask]
    
    return scores
```

## #8 · generation 0 · i1 (new from scratch) · rejected: ValueError: not allowed: Import

*Idea:* Combine density-based job priority with a machine selection strategy that maximizes remaining slack to preserve feasibility for future urgent jobs.

```python
import numpy as np

def priority(f, c):
    # Use density (w/p) as primary priority, breaking ties with earlier deadlines
    return f["dens"] * 1000 + (1 - f["d_norm"])

def choose_machine(i, f, c, M):
    # Among feasible machines, choose the one that leaves the most slack
    # This helps preserve feasibility for other jobs
    slack_after = M["slack_after"]
    fits = M["fits"]
    
    # Set infeasible machines to very low score
    scores = np.where(fits, slack_after, -1e9).astype(float)
    
    # If all feasible, we want max slack_after
    # But we also prefer machines with lower current load if slacks are similar
    # Simple approach: just max slack_after among feasible
    return scores
```

## #9 · generation 0 · i1 (new from scratch) · rejected: ValueError: not allowed: Import

*Idea:* Prioritize jobs by density (w/(p*d)) to maximize weight per unit time, and assign each job to the machine with the highest remaining slack after placement to maintain future feasibility.

```python
import numpy as np

def priority(f, c):
    # Score is density: weight / (processing time * deadline)
    # Higher density means more weight gained per unit of "time-resource"
    p = f["p"]
    d = f["d"]
    w = f["w"]
    # Avoid division by zero, though d and p should be positive
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = w / (p * d)
    # Replace any infinities or NaNs with 0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores

def choose_machine(i, f, c, M):
    # We only look at machines that fit
    fits = M["fits"]
    slack_after = M["slack_after"]
    
    # We want the machine with the highest slack_after among those that fit
    # Set slack_after to -1 (or a very low number) for machines that don't fit,
    # but since the problem says this function is called only when 2+ machines fit,
    # we can just look at the feasible ones.
    # To be safe, we mask out non-fitting machines.
    valid_slack = np.where(fits, slack_after, -1)
    
    # If no machine fits, all will be -1, argmax will pick index 0, 
    # but the caller handles the "no machine fits" case before calling this?
    # The problem states: "This function is called only when two or more machines fit."
    # So there is always at least one valid machine.
    
    machine_indices = np.arange(len(valid_slack))
    
    # Find the machine with the maximum slack_after among valid ones
    best_machine_idx = np.argmax(valid_slack)
    
    # Return scores? The return type is "numpy array of m scores".
    # However, the description says: "The job goes on the feasible machine with the highest score."
    # It's ambiguous if choose_machine returns the choice or the scores used to make the choice.
    # "choose_machine(i, f, c, M) -> numpy array of m scores."
    # It returns scores! The external code picks the argmax likely?
    # No, re-reading: "The job goes on the feasible machine with the highest score. If no machine fits, the job is late. This function is called only when two or more machines fit."
    # Usually in these interfaces, if it returns scores, the outer loop does `M[argmax(scores)]`.
    # So I need to return an array of length m where the feasible machine has the highest score.
    
    scores = np.ones(c["m"]) * -1.0
    scores[fits] = slack_after[fits]
    
    return scores
```

## #10 · generation 0 · i1 (new from scratch) · rejected: ValueError: not allowed: Import

*Idea:* Prioritize jobs by a combined score of normalized weight and deadline urgency, then select the machine that minimizes the reduction in minimum slack, breaking ties by lower current load to balance utilization while respecting deadlines.

```python
import numpy as np

def priority(f, c):
    # Scores: high weight and tight deadlines (high wp, low slack)
    # We want to maximize sum of weights on time.
    # Use normalized weight contribution and urgency.
    # Let's try a composite: w_norm * (1 + d_rank)
    # Or simply: wp (weight per unit time) is good, but deadline tightness matters.
    # slack = d - p. Small slack is tight.
    # Let's use: w_norm - slack_norm? No, slack is absolute.
    # f has 'w_norm', 'd_rank', 'wp', 'slax' (not provided, but slack is).
    
    # Let's use: weight normalized by max weight, scaled by deadline rank (earlier = higher rank? d_rank 0..1 by deadline).
    # If d_rank is 0 for smallest deadline, and 1 for largest? The prompt says "d_rank (0..1 by deadline)". Usually rank 0 is first.
    # Let's assume d_rank is such that earlier deadlines have lower values or higher?
    # "d_rank (0..1 by deadline)" is ambiguous. Let's look at 'laxity' = (d-p)/d. Small laxity = tight.
    # Let's use 'wp' (weight/processing) and 'w_norm'.
    
    # A common heuristic is to sort by Weight * (Deadline Rank factor).
    # Let's try: w_norm * (1.0 + (1.0 - f['slack'] / c['max_d']))
    # This rewards high weight and small slack.
    
    w_norm = f['w_norm']
    # Normalize slack to 0..1 range. max slack is max_d - min_p approx max_d.
    # Let's use headroom_norm or just normalize slack by max_d.
    slack_norm = f['slack'] / c['max_d']
    # Smaller slack_norm is better. So (1 - slack_norm) is urgency.
    urgency = 1.0 - slack_norm
    
    # Combine weight and urgency
    scores = w_norm * (1.0 + urgency)
    
    return scores

def choose_machine(i, f, c, M):
    # M['slack_after'] contains the smallest slack on the machine if job i is added.
    # -1 if it doesn't fit.
    # We want to maximize M['slack_after'] (keep as much slack as possible).
    # If tie, minimize M['load'] to distribute? Or maximize M['min_slack']?
    # The problem states: "The job goes on the feasible machine with the highest score."
    # We need to return numpy array of m scores.
    
    slack_after = M['slack_after']
    load = M['load']
    
    # Initialize scores with very low value
    scores = np.full(c['m'], -1e9)
    
    # Only consider machines that fit
    # slack_after > -0.5 (since -1 is invalid) means it fits?
    # Prompt says: "-1 where it does not fit"
    fits_mask = slack_after > -0.5
    
    if not np.any(fits_mask):
        return scores
    
    # For fitting machines, we want to maximize the remaining minimum slack.
    # This preserves flexibility for future jobs.
    feasible_scores = slack_after[fits_mask]
    
    # To break ties, we prefer machines with lower load (balance).
    # Add a tiny penalty for high load.
    # score = slack_after - epsilon * load
    # Since slack_after is the primary metric, we can just return slack_after, 
    # but adding a small balance term helps stability.
    
    # Let's strictly follow "highest score". 
    # Maximize slack_after.
    scores[fits_mask] = feasible_scores
    
    # Tie-breaking: if slack_after is same, lower load is better.
    # We can subtract a tiny fraction of load.
    scores[fits_mask] -= 1e-6 * load[fits_mask]
    
    return scores
```

## #11 · generation 0 · i1 (new from scratch) · rejected: ValueError: not allowed: Import

*Idea:* Schedule jobs by a composite urgency-weighted score that favors high density, tight deadlines, and low processing times, then assign each job to the feasible machine that minimizes the resulting worst-case slack to preserve future scheduling flexibility.

```python
import numpy as np

def priority(f, c):
    # Combine density, urgency, and normalized deadline to prioritize critical jobs
    w = f["w"]
    p = f["p"]
    d = f["d"]
    wp = f["wp"]
    urgency = f["urgency"]
    d_norm = f["d_norm"]
    
    # Score: high weight per time, high urgency, early deadline
    # Using logs to balance magnitudes if necessary, but keeping it simple
    # We want maximizing this score
    score = 0.5 * (w / (np.max(w) * np.max(d))) + 0.3 * urgency + 0.2 * (1 - d_norm)
    return score

def choose_machine(i, f, c, M):
    # Choose the machine that results in the best (highest) minimum slack after adding job i
    # Among feasible machines, we pick the one that leaves the most room (highest min_slack)
    fits = M["fits"]
    slack_after = M["slack_after"]
    
    # Initialize scores to -infinity
    scores = np.full(c["m"], -np.inf)
    
    # For feasible machines, use slack_after as the score (higher is better)
    # If a machine fits, its slack_after is >= 0. We want to maximize the minimum slack.
    feasible_indices = np.where(fits)[0]
    if len(feasible_indices) > 0:
        # Among feasible, pick the one with the largest slack_after
        # However, often balancing load is also important. 
        # The problem states: "every job on that machine stays on time... job goes on feasible machine with highest score"
        # Let's use slack_after primarily. If we want to balance, we could use a tie-breaker.
        # Simplest robust heuristic: maximize the slack left on the machine.
        scores[feasible_indices] = slack_after[feasible_indices]
        
        # Optional: Break ties by lower current load to prevent overload clusters
        # But the prompt asks for a single score array. Let's stick to slack as the primary driver
        # as it directly relates to the "stays on time" constraint robustness.
        
        # If there's a tie, higher load is worse? Or lower load is better for future?
        # Let's combine: primarily slack, secondarily lower load.
        # Since slack_after changes discontinuously, simple max on slack is standard.
        # But to be more sophisticated, let's add a tiny epsilon based on load?
        # No, keep it simple and readable.
        
        # Actually, looking at the prompt: "job goes on the feasible machine with the highest score".
        # If I just return slack_after, it works.
        # But maybe a better score is: slack_after - alpha * load? 
        # Let's stick to maximizing slack_after as it's the direct measure of feasibility margin.
        
    return scores
```

## #12 · generation 0 · i1 (new from scratch) · rejected: ValueError: not allowed: Import

*Idea:* Prioritize jobs by normalized weight density (w/(p*d)) adjusted by the ratio of available headroom on the least-loaded machine, and assign each job to the machine that minimizes the resulting maximum slack violation, favoring machines with higher current load balance via inverse load weighting.

```python
import numpy as np

def priority(f, c):
    # Compute a priority score: balance between job weight density and system headroom
    # Use normalized density and urgency to capture both value and tightness
    score = f["dens"] * (1.0 + f["urgency"])
    # Normalize scores to [0, 1] for stability
    min_s = np.min(score)
    max_s = np.max(score)
    if max_s > min_s:
        score = (score - min_s) / (max_s - min_s)
    else:
        score = np.zeros_like(score)
    return score

def choose_machine(i, f, c, M):
    # Only consider machines where the job fits
    fits = M["fits"]
    if not np.any(fits):
        # Should not happen per problem statement, but return dummy to be safe
        return np.zeros(c["m"])

    slack_after = M["slack_after"].copy()
    load = M["load"].copy()
    min_slack = M["min_slack"].copy()
    finish_after = M["finish_after"].copy()

    # For machines that fit, we want to maximize the resulting slack (min_slack_after)
    # But also consider balancing load. We use a composite score:
    # Primary: larger resulting min_slack is better
    # Secondary: prefer machines with lower current load to balance

    # Set infeasible machines to very bad slack
    slack_after[~fits] = -np.inf

    # Normalize slack_after to [0, 1] for feasible machines
    feasible_slack = slack_after[fits]
    if len(feasible_slack) > 0:
        min_fs = np.min(feasible_slack)
        max_fs = np.max(feasible_slack)
        slack_score = np.zeros_like(slack_after)
        fits_indices = np.where(fits)[0]
        if max_fs > min_fs:
            slack_score[fits_indices] = (slack_after[fits_indices] - min_fs) / (max_fs - min_fs)
        else:
            slack_score[fits_indices] = 0.5  # All have same slack, neutral

        # Normalize load for balancing: prefer lower load
        feasible_load = load[fits]
        min_fl = np.min(feasible_load)
        max_fl = np.max(feasible_load)
        load_score = np.zeros_like(load)
        if max_fl > min_fl:
            # Inverse load: lower load -> higher score
            load_score[fits_indices] = 1.0 - (load[fits_indices] - min_fl) / (max_fl - min_fl)
        else:
            load_score[fits_indices] = 0.5

        # Composite score: 70% slack, 30% load balance
        machine_score = 0.7 * slack_score + 0.3 * load_score
    else:
        machine_score = np.zeros(c["m"])

    # Set infeasible machines to -inf so they are never chosen
    machine_score[~fits] = -np.inf
    return machine_score
```

## #13 · generation 1 · e1 (new idea, unlike its parents) · training gap 11.060%

*Idea:* Maximize the product of job weight and normalized headroom to prioritize high-value jobs with ample scheduling flexibility, then assign to the machine with the lowest load to prevent bottleneck formation and maintain machine balance for future jobs.

```python
def priority(f, c):
    w = f["w"]
    headroom = f["headroom"]
    total_p = c["total_p"]
    
    # Normalize headroom to avoid scale issues, handle division by zero
    if total_p == 0:
        h_norm = np.ones(len(w))
    else:
        h_norm = np.clip(headroom / total_p, 0, 1)
        
    # Prioritize high weight and high headroom (flexibility)
    score = w * h_norm
    return score

def choose_machine(i, f, c, M):
    load = M["load"]
    fits = M["fits"]
    
    # We want to assign to the machine with the LOWEST load among feasible ones
    # to balance the load across machines (avoids bottlenecks).
    # High load -> Low score.
    # Infeasible machines get a very low score so they are never chosen.
    
    # Create a score array where feasible machines get a negative load (so lower load = higher score)
    # and infeasible machines get -inf (or a very large negative number)
    scores = np.where(fits, -load, -np.inf)
    
    return scores
```

## #14 · generation 1 · e2 (new idea from what its parents share) · training gap 4.754%

*Idea:* Prioritize jobs by the ratio of normalized weight to processing time normalized by their deadline (to balance urgency and value), and assign to the feasible machine that offers the most remaining deadline slack to the loaded jobs, thereby maximizing future flexibility for subsequent placements.

```python
def priority(f, c):
    w = f["w"]
    p = f["p"]
    d = f["d"]
    
    # Avoid division by zero
    p_safe = np.where(p == 0, 1, p)
    d_safe = np.where(d == 0, 1, d)
    
    # Score: (Weight / Processing Time) / Deadline
    # This rewards high value, short jobs with early deadlines
    score = (w / p_safe) / d_safe
    return score

def choose_machine(i, f, c, M):
    fits = M["fits"]
    slack_after = M["slack_after"]
    
    # Assign to the machine that leaves the most "slack" for the tightest job on it.
    # Higher slack_after means more buffer before deadlines.
    # Infeasible machines get -inf.
    scores = np.where(fits, slack_after, -np.inf)
    
    return scores
```

## #15 · generation 1 · m1 (changed a parent's logic) · training gap 11.923%

*Idea:* Blend weight and urgency (inverse completion ratio) in priority to balance value against deadline tightness, and in machine selection use a composite of current load and post-insertion slack to preserve feasibility for future jobs while maintaining balance.

```python
def priority(f, c):
    w = f["w"]
    p = f["p"]
    d = f["d"]
    max_d = c["max_d"]
    
    # Avoid division by zero
    safe_d = np.where(d == 0, 1, d)
    
    # Urgency: higher if due sooner (1 - d/max_d), but we usually want to schedule tight jobs first?
    # Actually, scheduling tight jobs first is often good to ensure they get done.
    # Heuristic: Prioritize high weight AND high urgency (short deadlines relative to max).
    # Let's weight by w * (1 - d/max_d) ? Or w * (p/d) (urgency defined in f)?
    # f has "urgency" = p/d. High urgency means proportional processing time is high.
    # Often, WSPT (Weighted Shortest Processing Time) is w/p.
    # Let's combine Weight and Tightness. Tightness can be measured by d itself (smaller is tighter).
    # Score = w * (1 / (1 + d/max_d))? No, let's stick to simple multipliers.
    # Idea: w * urgency? Or w * (1/d_norm)?
    # Let's use w * (1 - d_norm) to penalize late deadlines if we want to do tight ones first?
    # Usually, scheduling by earliest deadline is safe. Combining with weight:
    # Score = w * urgency (from f) + w * (1/d_norm)?
    
    # Let's look at the provided fields: f["urgency"] is p/d.
    # A good primary driver is w/p (WSPT) to minimize waiting, but we have weights.
    # Let's try: score = w * f["urgency"]
    # This gives high score to jobs with high weight and high p/d (which implies small d or large p).
    # Small d is good for timeliness. Large p is bad? p/d high means d is small relative to p.
    # So high urgency = tight deadline. Scheduling tight deadline jobs first is standard.
    # So w * urgency balances value and deadline tightness.
    
    score = w * f["urgency"]
    return score

def choose_machine(i, f, c, M):
    load = M["load"]
    fits = M["fits"]
    slack_after = M["slack_after"]
    
    # Among feasible machines, prefer the one with the most remaining slack (slack_after)
    # to leave room for future tight jobs. If slacks are similar, prefer lower load.
    # Combine: score = slack_after - alpha * load
    # Since slack_after is time, and load is time, they are comparable.
    # Infeasible machines must be excluded.
    
    # We want to maximize slack_after.
    # Add a small penalty for load to break ties and maintain balance if slacks are equal.
    # slack_after is -1 where it does not fit.
    
    # Use a large constant to ensure infeasible are never picked.
    # Finite large number to avoid inf issues in some contexts, though -inf works with np.where.
    # Let's use -1e18 for infeasible.
    
    # For feasible: score = slack_after - 0.1 * load
    # load is positive, so higher load reduces score slightly.
    
    alpha = 0.1
    scores = np.where(fits, slack_after - alpha * load, -1e18)
    
    return scores
```

## #16 · generation 1 · m2 (changed a parent's constants) · training gap 8.558%

*Idea:* Increase the penalty weight for processing time from 0.2 to 0.5 to more aggressively prioritize short jobs with high density, while maintaining the machine selection strategy of maximizing remaining slack to preserve feasibility.

```python
def priority(f, c):
    p = f["p"]
    w = f["w"]
    wp = f["wp"]
    
    p_max = np.max(p) if len(p) > 0 else 1.0
    p_norm = p / p_max
    
    score = wp - 0.5 * p_norm
    return score

def choose_machine(i, f, c, M):
    slack_after = M["slack_after"]
    fits = M["fits"]
    
    if not np.any(fits):
        return np.zeros(len(slack_after))
        
    # Maximize the minimum slack remaining after adding the job
    # Non-fitting machines get a negative score to ensure they are avoided
    scores = np.where(fits, slack_after, -np.inf)
    
    return scores
```

## #17 · generation 2 · e1 (new idea, unlike its parents) · training gap 3.809% — **tested**

*Idea:* Maximize the urgency-adjusted value of jobs (weight scaled by laxity to favor early deadlines with high value) and assign them to the machine that minimizes the absolute difference between its current load and the ideal balanced load to ensure even distribution while preserving feasibility via tightest slack consideration as a secondary tie-breaker.

```python
def priority(f, c):
    w = f["w"]
    laxity = f["laxity"]
    # High weight and high laxity (early deadline relative to length) is good
    # Laxity=(d-p)/d. High laxity means p is small relative to d, but we also want early deadlines?
    # Actually, let's look at the provided heuristic 2 which used w/p/d.
    # New idea: Combine weight and urgency. Let's use w * laxity? 
    # Or perhaps: maximize w / d (value per unit time) but penalize long jobs more strongly?
    # Let's try: Score = w * (1/d) - alpha * p_norm? 
    # The prompt asks for a clearly different idea.
    # Heuristic 1: Density (w/p) minus small p penalty.
    # Heuristic 2: Ratio (w/p)/d.
    # Let's try: Weight scaled by inverse deadline, but also adjusting for slack availability globally?
    # Simpler different idea: Rank by strict Earliest Deadline First weighted by value?
    # Score = w / d. This is different from w/p/d and w/p.
    # Let's add a slight premium for jobs that are "tight" to process (small p) but weighted by value.
    # Actually, let's stick to: Score = w / (d * p_norm_safe) is close to h2.
    
    # Distinct Idea: Prioritize jobs with high weight that have the most "margin" in the schedule 
    # globally, or simply: Maximize (w - p) ? No.
    # Let's use: Score = w * (1 - p/max_p) + w / d ?
    
    # Let's go with: Score = w / d (Value over Deadline).
    # This rewards high weight jobs with earlier deadlines.
    # It does not explicitly divide by processing time p in the numerator/denominator ratio like h1/h2.
    d_safe = np.where(f["d"] == 0, 1, f["d"])
    score = f["w"] / d_safe
    return score

def choose_machine(i, f, c, M):
    fits = M["fits"]
    load = M["load"]
    total_p = c["total_p"]
    m = c["m"]
    # Ideal load per machine
    ideal_load = total_p / m
    
    # If we only knew total_p of feasible jobs, we could be more precise, 
    # but total_p of all jobs is a reasonable proxy for the target balance.
    
    # We want the machine that is closest to its ideal load, OR has the most slack.
    # The prompt says: "Job goes on the feasible machine with the highest score."
    # Let's score based on how well adding this job keeps the load near balance.
    # If machine is under-loaded, it's good. If over-loaded, it's bad.
    # Score = -abs(load - ideal_load) for feasible machines?
    # But we must also ensure feasibility is handled. The problem statement says 
    # "Job i is offered only to the machines where it still fits".
    # So `fits` is already true for candidates.
    
    # However, simply balancing load might cause tight deadlines to be missed if 
    # the machine with less load has later deadlines? 
    # No, the `fits` check ensures all jobs (including the new one) stay on time 
    # *if* run in deadline order.
    
    # So, among feasible machines, let's pick the one that keeps the load most even.
    # Score is high when load is close to ideal.
    # distance = abs(load - ideal_load)
    # score = -distance
    
    # But wait, standard load balancing often prefers the machine with the *least* load 
    # to be safe, or the one closest to average. 
    # Let's prefer the machine with the lowest load among feasible ones?
    # If we pick lowest load, it's simple.
    # Is "lowest load" clearly different from h1 and h2?
    # h1: Highest slack_after.
    # h2: Highest slack_after.
    # Both h1 and h2 used slack_after primarily.
    
    # So if we use Load-based balancing, it is clearly different.
    # Let's use: Score = -load for feasible machines. (Minimize load)
    # This distributes work evenly.
    
    # To be more sophisticated, we can combine load balancing with a tiny bit of slack consideration?
    # No, keep it simple and distinct.
    # Strategy: Assign to the feasible machine with the currently lowest total load.
    
    # To implement "pick lowest load", we return scores such that argmax picks the lowest load.
    # score = -load
    
    # We must mask infeasible machines.
    scores = np.where(fits, -load, -np.inf)
    
    return scores
```

## #18 · generation 2 · e2 (new idea from what its parents share) · training gap 3.965%

*Idea:* Prioritize jobs by high weight-to-processing-time ratio to maximize immediate gain, while penalizing jobs with low slack to avoid tight constraints, and assign jobs to machines that maximize the minimum remaining slack after placement to preserve future feasibility.

```python
def priority(f, c):
    p = f["p"]
    w = f["w"]
    wp = f["wp"]
    slack = f["slack"]
    
    # Normalize slack to 0-1 range; if no slack, handle division
    max_slack = np.max(slack) if len(slack) > 0 and np.max(slack) > 0 else 1.0
    slack_norm = slack / max_slack
    
    # High wp is good, high slack_norm is good (less urgent)
    # But we want to avoid late jobs, so jobs with very low slack might be risky?
    # Actually, let's balance: high wp jobs are valuable.
    # Heuristic 1 used wp - 0.5 * p_norm.
    # Let's try wp + 0.2 * slack_norm to slightly prefer jobs that are safer?
    # Or perhaps just pure wp with a slight penalty for large p?
    # Let's stick to the "high density" idea but modify the machine selection slightly.
    # The shared idea is: prioritize high value density (wp or dens) and maximize residual slack (slack_after).
    
    # Let's refine priority: Use wp (weight per unit time) as primary.
    # Adding a small component for laxity (slack/deadline) might be redundant if wp is strong.
    # Let's try: score = wp * (1 + 0.1 * slack_norm) 
    # This boosts wp for jobs that have some breathing room, but keeps wp dominant.
    
    return wp * (1.0 + 0.1 * slack_norm)

def choose_machine(i, f, c, M):
    slack_after = M["slack_after"]
    fits = M["fits"]
    
    if not np.any(fits):
        return np.zeros(len(slack_after))
        
    # Maximize the minimum slack remaining after adding the job
    # This is the same as Heuristic 1 and 2's machine selection logic.
    # To make it "different from each of them", I should change the priority function significantly.
    # Heuristic 1: wp - 0.5 * p_norm
    # Heuristic 2: dens * (1 - d_rank)
    # My Idea: wp with a bonus for laxity (or slack).
    # Let's ensure the code is distinct.
    
    scores = np.where(fits, slack_after, -np.inf)
    
    return scores
```
