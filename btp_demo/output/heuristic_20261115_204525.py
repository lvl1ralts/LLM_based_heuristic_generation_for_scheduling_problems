"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 2s)

Idea: Prioritize jobs by weight‑per‑time boosted by how lax they are, and place each job on the feasible machine that keeps the largest slack while also staying light on load.
"""
import math

import numpy as np


def priority(f, c):
    # high wp (weight/processing) and high laxity (more slack) are desirable
    return f["wp"] * f["laxity"]


def choose_machine(i, f, c, M):
    # raw slack after adding the job
    slack = M["slack_after"]
    # projected load on the machine if the job is placed there
    proj_load = M["load"] + f["p"][i]
    # score = slack minus a small penalty proportional to the resulting load
    score = slack - proj_load / c["max_d"]
    # infeasible machines get worst possible score
    score = np.where(M["fits"], score, -np.inf)
    return score
