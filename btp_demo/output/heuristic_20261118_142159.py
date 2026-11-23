"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 2s)

Idea: Combine weighted‑shortest‑processing time with tightness (low laxity) for ordering, and pick the feasible machine that keeps the largest slack while also preferring a less‑loaded machine to retain future capacity.
"""
import math

import numpy as np


def priority(f, c):
    # high weight/processing, but give extra priority to jobs with little laxity (tight deadlines)
    # also add a tiny component of density to break ties favorably
    return f["wp"] * (1.0 - f["laxity"]) + 0.001 * f["dens"]


def choose_machine(i, f, c, M):
    # score = slack after placement, penalised slightly by current load
    # (larger slack is better, smaller load is better)
    return M["slack_after"] - 0.001 * M["load"]
