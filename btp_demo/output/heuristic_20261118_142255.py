"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 2s)

Idea: Combine weight‑per‑time with deadline tightness for priority and select the feasible machine that maximizes slack after placement while penalizing already‑loaded machines.
"""
import math

import numpy as np


def priority(f, c):
    # high wp and low laxity (tight deadlines) get higher scores
    return f["wp"] * (1.0 + (1.0 - f["laxity"]))


def choose_machine(i, f, c, M):
    # score = slack after placement divided by current load (plus tiny epsilon)
    eps = 1e-9
    return M["slack_after"] / (M["load"] + eps)
