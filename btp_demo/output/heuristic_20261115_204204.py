"""Heuristic for Pm | r=0 | sum w U  (live: groq:openai/gpt-oss-120b, answered in 3s)

Idea: Combine weight‑per‑processing with job laxity for priority and, when several machines can host a job, choose the one with the largest slack after placement while slightly penalising heavily loaded machines.
"""
import math

import numpy as np


def priority(f, c):
    # Boost WSPT (weight per processing) by the job's laxity (more slack → higher score)
    # laxity = (d - p) / d ; already provided in f
    return f["wp"] * (1.0 + f["laxity"])


def choose_machine(i, f, c, M):
    # Base score = slack remaining after adding the job
    # Penalise machines that are already heavily loaded to improve balance
    # Small factor (0.01) scales the load penalty relative to slack magnitudes
    return M["slack_after"] - 0.01 * M["load"]
