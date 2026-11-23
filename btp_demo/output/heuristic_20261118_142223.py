"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 2s)

Idea: Combine weight‑per‑time with deadline urgency for priority, and pick the feasible machine that maximizes slack while penalizing already‑loaded machines.
"""
import math

import numpy as np


def priority(f, c):
    # higher weight per processing time, boosted for tight deadlines (urgency = p/d)
    return f["wp"] * (1.0 + f["urgency"])


def choose_machine(i, f, c, M):
    # normalized load to keep machines balanced
    norm_load = M["load"] / (c["max_d"] * c["m"])
    # score = remaining slack minus load penalty
    score = M["slack_after"] - norm_load
    # infeasible machines get -inf so they are never chosen
    score = np.where(M["fits"], score, -np.inf)
    return score
