"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 3s)

Idea: Combine weight‑per‑processing with urgency to rank jobs and select the feasible machine that maximizes slack while penalising heavily loaded machines.
"""
import math

import numpy as np


def priority(f, c):
    # Favor high weight per processing and urgent (large p/d) jobs
    return f["wp"] * (1.0 + f["urgency"])


def choose_machine(i, f, c, M):
    # Penalise machines that are already heavily loaded.
    # load is normalized by the maximum possible load per machine (max deadline)
    load_norm = M["load"] / c["max_d"]
    # Score: high remaining slack and low load
    return M["slack_after"] * (1.0 - load_norm)
