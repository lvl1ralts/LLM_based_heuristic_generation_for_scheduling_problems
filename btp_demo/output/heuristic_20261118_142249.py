"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 2s)

Idea: Combine weight‑per‑processing with urgency for priority and choose the machine that maximizes slack after while penalizing already loaded machines, thus balancing flexibility and load.
"""
import math

import numpy as np


def priority(f, c):
    # high wp jobs that are also urgent (short deadline) get higher scores
    return f["wp"] * (1.0 + f["urgency"])


def choose_machine(i, f, c, M):
    # score = slack after / (1 + normalized load)
    # higher slack is good, but heavily loaded machines are penalized
    load_norm = M["load"] / c["max_d"]            # normalise load
    scores = M["slack_after"] / (1.0 + load_norm) # higher is better
    # ensure non‑feasible machines are never chosen
    scores = np.where(M["fits"], scores, -np.inf)
    return scores
