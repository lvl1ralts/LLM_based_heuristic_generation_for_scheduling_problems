"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 2s)

Idea: Combine weight‑per‑time with urgency to prioritize high‑value tight jobs, and select a machine by rewarding large remaining slack while penalizing high current load for better balance.
"""
import math

import numpy as np


def priority(f, c):
    # high weight per time, higher when the job is urgent (large p/d), plus density
    return f["wp"] * (1.0 + f["urgency"]) + f["dens"]


def choose_machine(i, f, c, M):
    # normalize slack by the maximum deadline
    norm_slack = np.where(M["slack_after"] >= 0,
                         M["slack_after"] / c["max_d"],
                         -np.inf)
    # normalize load by the average load per machine
    avg_load = c["total_p"] / c["m"]
    load_penalty = M["load"] / avg_load
    # higher score = large slack, low load
    return norm_slack - load_penalty
