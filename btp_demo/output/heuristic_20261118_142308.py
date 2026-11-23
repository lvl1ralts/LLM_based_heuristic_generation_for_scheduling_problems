"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 2s)

Idea: Combine the WSPT priority with a machine‑selection score that balances the resulting slack and current load, preferring machines that keep high slack while staying load‑balanced.
"""
import math

import numpy as np


def priority(f, c):
    # keep the effective WSPT order
    return f["wp"]


def choose_machine(i, f, c, M):
    # score = slack after placement, penalised by current load
    # non‑feasible machines have slack_after = -1, give them -inf score
    slack = M["slack_after"]
    load = M["load"]
    score = np.where(slack > -1,
                     slack * 1.0 / (1.0 + load),   # higher slack, lower load = better
                     -np.inf)
    return score
