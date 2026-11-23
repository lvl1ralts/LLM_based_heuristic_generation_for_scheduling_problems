"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 2s)

Idea: Combine weight‑per‑time with deadline urgency for priority and pick the feasible machine that keeps the most slack while also avoiding overload, i.e., high slack – scaled load.
"""
import math

import numpy as np


def priority(f, c):
    # high weight per processing time, boosted for jobs that are urgent (large p/d)
    return f["wp"] * (1.0 + f["urgency"])


def choose_machine(i, f, c, M):
    # slack after adding the job (higher is better)
    slack = M["slack_after"]
    # projected load on each machine if the job were placed there
    load_after = M["load"] + f["p"][i]
    # normalise load by the maximum deadline to keep scale comparable
    load_norm = load_after / c["max_d"]
    # score = slack – α·load_norm (α balances the two criteria)
    score = slack - 0.5 * load_norm
    # infeasible machines have slack = -1; give them -inf so they are never chosen
    score = np.where(slack >= 0, score, -np.inf)
    return score
