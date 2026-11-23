"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 3s)

Idea: Combine weight‑per‑processing with deadline urgency for job ordering, and pick the feasible machine that maximizes a slack‑adjusted metric that also favours less‑loaded machines.
"""
import math

import numpy as np


def priority(f, c):
    # high weight per time and tight deadlines get higher scores
    return f["wp"] * f["urgency"]


def choose_machine(i, f, c, M):
    # effective slack = slack after job / (current load + job processing)
    p_i = f["p"][i]
    load_plus = M["load"] + p_i
    # avoid division by zero
    eps = 1e-9
    eff_slack = M["slack_after"] / (load_plus + eps)
    # machines where the job does not fit get the worst possible score
    return np.where(M["fits"], eff_slack, -np.inf)
