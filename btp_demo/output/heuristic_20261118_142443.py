"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 2s)

Idea: Combine weight‑per‑processing with urgency for priority, and pick the feasible machine that maximizes slack after while preferring less‑loaded machines.
"""
import math

import numpy as np


def priority(f, c):
    # higher weight per time, boosted for tighter (high urgency) jobs
    return f["wp"] * (2.0 - f["urgency"])


def choose_machine(i, f, c, M):
    # slack after job i on each machine (‑1 if infeasible)
    slack = M["slack_after"]
    load = M["load"]
    # score = slack * (1 / (1+load)) → prefer larger slack and smaller load
    score = slack * (1.0 / (1.0 + load))
    # infeasible machines get a very low score
    return np.where(slack >= 0, score, -np.inf)
