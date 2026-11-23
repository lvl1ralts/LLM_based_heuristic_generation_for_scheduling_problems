"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 2s)

Idea: Combine weight‑per‑processing with laxity for priority, and when several machines can accept the job, pick the one that keeps the largest slack while also preferring a less‑loaded machine.
"""
import math

import numpy as np


def priority(f, c):
    # high wp (weight per time) and high laxity (more slack) are good
    # scale laxity to [0,1] already; add it as a multiplier
    return f["wp"] * (1.0 + f["laxity"])

def choose_machine(i, f, c, M):
    # score = slack after placement minus a small load penalty
    # normalize load by the maximum possible load per machine (max_d * m)
    norm_load = M["load"] / (c["max_d"] * c["m"])
    return M["slack_after"] - norm_load
