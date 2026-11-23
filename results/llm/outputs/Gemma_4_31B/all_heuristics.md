# Every heuristic Gemma 4 31B wrote
Called through Google Gemini API (Gemini key); open-weights model. 24 heuristics written, 21 ran correctly. Training gap = mean % gap to the LP bound on the 12 training instances (lower is better). WSPT scores 4.183%, the hand-written seed 3.503%.
The one marked **tested** has the lowest training gap among the model's own heuristics and is the one run in report 14.

## #0 · generation 0 · seed (hand-written start (WSPT, first fit)) · training gap 3.503%

*Idea:* WSPT priority (weight per unit of processing time); the job goes on the first machine where it fits.

```python
def priority(f, c):
    return f["wp"]


def choose_machine(i, f, c, M):
    return -np.arange(len(M["fits"]))
```

## #1 · generation 0 · i1 (new from scratch) · training gap 4.849%

*Idea:* Prioritize jobs with high weight-to-processing-time ratio and early deadlines, then assign them to the machine that preserves the maximum minimum slack.

```python
def priority(f, c):
    return f["wp"] / (f["d_norm"] + 0.1)

def choose_machine(i, f, c, M):
    return M["slack_after"]
```

## #2 · generation 0 · i1 (new from scratch) · training gap 5.936%

*Idea:* Prioritize jobs with high weight-to-processing-time ratios and early deadlines, assigning them to the machine that retains the most slack to maintain overall flexibility.

```python
def priority(f, c):
    return f["wp"] / (f["d_rank"] + 0.1)

def choose_machine(i, f, c, M):
    return M["slack_after"]
```

## #3 · generation 0 · i1 (new from scratch) · training gap 4.110%

*Idea:* Prioritize jobs based on a combination of weight-to-processing-time ratio and urgency, then assign each job to the machine that preserves the maximum minimum slack.

```python
def priority(f, c):
    # Combine weight density (w/p) with urgency (p/d) to prioritize high-value, 
    # tight-deadline jobs.
    return f["wp"] * (1.0 + f["urgency"])

def choose_machine(i, f, c, M):
    # Maximize the minimum slack on the machine after adding the job
    # to keep the machine as flexible as possible for future jobs.
    return M["slack_after"]
```

## #4 · generation 0 · i1 (new from scratch) · training gap 4.110%

*Idea:* Prioritize jobs by weight density adjusted for urgency, then place each job on the machine that maintains the most flexibility.

```python
def priority(f, c):
    return f["wp"] * (1 + f["urgency"])

def choose_machine(i, f, c, M):
    return M["slack_after"]
```

## #5 · generation 0 · i1 (new from scratch) · training gap 4.110%

*Idea:* Prioritize jobs by combining weight density and urgency, then assign them to the machine that preserves the most remaining slack.

```python
def priority(f, c):
    # Score combines value density (w/p) and absolute urgency (w/d)
    return f["wp"] * (1.0 + f["urgency"])

def choose_machine(i, f, c, M):
    # Use a 'Worst Fit' approach by maximizing the remaining slack on the machine
    return M["slack_after"]
```

## #6 · generation 1 · e1 (new idea, unlike its parents) · training gap 4.355%

*Idea:* Prioritize jobs by weighting their value density against the available headroom of their deadline window and assign them to the machine that minimizes the job's finish time.

```python
def priority(f, c):
    # Prioritize jobs with high value density and low headroom (more constrained windows)
    return f["wp"] * (1.0 - f["headroom_norm"])

def choose_machine(i, f, c, M):
    # Assign the job to the machine that completes it earliest to preserve future space
    return -M["finish_after"]
```

## #7 · generation 1 · e2 (new idea from what its parents share) · training gap 4.957%

*Idea:* Prioritize jobs by weight density and assign them to the machine that balances remaining slack with current load to preserve flexibility.

```python
def priority(f, c):
    return f["dens"]

def choose_machine(i, f, c, M):
    return M["slack_after"] - M["load"] / c["max_d"]
```

## #8 · generation 1 · m1 (changed a parent's logic) · training gap 5.002%

*Idea:* Prioritize jobs with high weight-to-processing-time ratios and early deadlines, assigning them to the most loaded machine that still fits to preserve empty space on other machines.

```python
def priority(f, c):
    return f["wp"] / (f["d_rank"] + 0.1)

def choose_machine(i, f, c, M):
    return M["load"]
```

## #9 · generation 1 · m2 (changed a parent's constants) · training gap 4.678%

*Idea:* Prioritize jobs by increasing the penalty for available headroom using a power law to further favor highly constrained, high-density jobs.

```python
def priority(f, c):
    # Prioritize jobs with high value density and strongly penalize those with more headroom
    return f["wp"] * (1.0 - f["headroom_norm"])**2

def choose_machine(i, f, c, M):
    # Assign the job to the machine that completes it earliest to preserve future space
    return -M["finish_after"]
```

## #10 · generation 2 · e1 (new idea, unlike its parents) · API call failed, nothing written: RuntimeError: <urlopen error [Errno 11001] getaddrinfo failed>

## #11 · generation 2 · e2 (new idea from what its parents share) · API call failed, nothing written: RuntimeError: <urlopen error [Errno 11001] getaddrinfo failed>

## #12 · generation 2 · m1 (changed a parent's logic) · training gap 4.147%

*Idea:* Use WSPT priority to focus on high-weight, short-processing jobs, and assign them to the machine that maximizes the resulting minimum slack to preserve future flexibility.

```python
def priority(f, c):
    return f["wp"]


def choose_machine(i, f, c, M):
    return M["slack_after"]
```

## #13 · generation 2 · m2 (changed a parent's constants) · training gap 4.258%

*Idea:* Prioritize jobs with a higher weight-to-processing-time ratio and a stronger emphasis on early deadlines, while continuing to maximize the minimum remaining slack on the chosen machine.

```python
def priority(f, c):
    return f["wp"] / (f["d_norm"]**0.5 + 0.01)

def choose_machine(i, f, c, M):
    return M["slack_after"]
```

## #14 · generation 3 · e1 (new idea, unlike its parents) · training gap 4.364%

*Idea:* Prioritize jobs using the weight-to-processing-time ratio scaled by an exponential of the system-wide congestion (headroom) at their deadline, while balancing the load across machines to preserve future flexibility.

```python
def priority(f, c):
    return f["wp"] * np.exp(-f["headroom_norm"])

def choose_machine(i, f, c, M):
    return -M["load"]
```

## #15 · generation 3 · e2 (new idea from what its parents share) · training gap 4.094%

*Idea:* Prioritize jobs by balancing their weight-to-processing-time ratio against their deadline urgency, and assign them to the machine that maximizes the remaining minimum slack to maintain overall flexibility.

```python
def priority(f, c):
    return f["wp"] / (f["d_norm"] + 0.5)

def choose_machine(i, f, c, M):
    return M["slack_after"]
```

## #16 · generation 3 · m1 (changed a parent's logic) · training gap 3.470% — **tested**

*Idea:* Use WSPT priority to focus on high-value jobs and a Best-Fit machine selection strategy by maximizing the load on the chosen machine to preserve large contiguous time blocks on other machines.

```python
def priority(f, c):
    return f["wp"]


def choose_machine(i, f, c, M):
    return M["load"]
```

## #17 · generation 3 · m2 (changed a parent's constants) · rejected: ValueError: duplicate of an earlier heuristic

## #18 · generation 4 · e1 (new idea, unlike its parents) · training gap 4.857%

*Idea:* Prioritize jobs by their weight-to-area density (weight divided by the product of processing time and deadline) and assign each job to the machine that allows it to finish earliest.

```python
def priority(f, c):
    return f["dens"]

def choose_machine(i, f, c, M):
    return -M["finish_after"]
```

## #19 · generation 4 · e2 (new idea from what its parents share) · training gap 3.497%

*Idea:* Use WSPT priority to focus on high-value, short jobs and select the machine that maximizes the resulting slack relative to the current load to balance flexibility and packing.

```python
def priority(f, c):
    return f["wp"]


def choose_machine(i, f, c, M):
    return M["slack_after"] / (M["load"] + 1)
```

## #20 · generation 4 · m1 (changed a parent's logic) · training gap 4.277%

*Idea:* Prioritize jobs using a weighted combination of their weight-to-processing-time ratio and the inverse of their normalized deadline to balance high value with urgency, then assign them to the machine that maximizes the resulting minimum slack.

```python
def priority(f, c):
    # Prioritize jobs with high weight density (w/p) and earlier deadlines.
    # The 1.0 + (1.0 - f["d_rank"]) factor ensures w/p remains dominant while 
    # giving a significant boost to more urgent jobs.
    return f["wp"] * (2.0 - f["d_rank"])

def choose_machine(i, f, c, M):
    # Maximize the minimum slack on the machine after adding the job.
    # This preserves the most flexibility for future jobs.
    return M["slack_after"]
```

## #21 · generation 4 · m2 (changed a parent's constants) · training gap 3.944%

*Idea:* Increase the weight of the urgency component in the priority score to more aggressively prioritize jobs with tighter deadlines while maintaining the max-min slack machine assignment.

```python
def priority(f, c):
    # Increase the urgency multiplier to better balance weight density and deadline pressure
    return f["wp"] * (1.0 + 2.0 * f["urgency"])

def choose_machine(i, f, c, M):
    # Maintain the logic of maximizing the minimum slack to preserve flexibility
    return M["slack_after"]
```

## #22 · generation 5 · e1 (new idea, unlike its parents) · training gap 4.301%

*Idea:* Prioritize jobs with the highest weight-to-processing-time-deadline density and assign them to the most loaded feasible machine to preserve larger contiguous blocks of time on other machines.

```python
def priority(f, c):
    # Prioritize jobs with high value relative to their processing time and deadline
    return f["dens"]

def choose_machine(i, f, c, M):
    # Use a "Best Fit" strategy by assigning the job to the most loaded machine
    # that can still accommodate it, preserving capacity on less loaded machines.
    return M["load"]
```

## #23 · generation 5 · e2 (new idea from what its parents share) · training gap 5.388%

*Idea:* Prioritize jobs by balancing their value density with their deadline urgency and assign them to machines by maximizing the ratio of remaining slack to current load.

```python
def priority(f, c):
    # Prioritize jobs with high weight-to-processing ratio and early deadline rank
    return f["wp"] / (f["d_rank"] + 0.1)

def choose_machine(i, f, c, M):
    # Favor machines that maintain high slack while having lower overall load
    return M["slack_after"] / (M["load"] + 1)
```

## #24 · generation 5 · m1 (changed a parent's logic) · rejected: ValueError: duplicate of an earlier heuristic

## #25 · generation 5 · m2 (changed a parent's constants) · API call failed, nothing written: RemoteDisconnected: Remote end closed connection without response

## #26 · generation 6 · e1 (new idea, unlike its parents) · API call failed, nothing written: RemoteDisconnected: Remote end closed connection without response

## #27 · generation 6 · e2 (new idea from what its parents share) · API call failed, nothing written: RuntimeError: HTTP 500: {
  "error": {
    "code": 500,
    "message": "Internal

## #28 · generation 6 · m1 (changed a parent's logic) · training gap 4.219%

*Idea:* Use a hybrid priority that balances weighted processing time (WSPT) and deadline urgency (EDD) using the square root of the normalized deadline, and select the machine that maximizes the resulting minimum slack to preserve future flexibility.

```python
def priority(f, c):
    return f["wp"] / np.sqrt(f["d_norm"])


def choose_machine(i, f, c, M):
    return M["slack_after"]
```

## #29 · generation 6 · m2 (changed a parent's constants) · rejected: ValueError: duplicate of an earlier heuristic
