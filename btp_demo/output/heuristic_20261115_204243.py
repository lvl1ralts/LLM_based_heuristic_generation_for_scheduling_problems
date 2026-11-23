"""Heuristic for Pm | r=0 | sum w U  (live: groq:qwen/qwen3.8-27b, answered in 1s)

Idea: Use a priority that weights jobs by their weight normalized by processing time and deadline to favor urgent, high-value jobs, and assign them to the machine that maximizes the minimum slack after job placement to preserve future scheduling flexibility.
"""
import math

import numpy as np


def priority(f, c):
    # Weighted Shortest Processing Time adjusted for urgency
    # Use wp = w/(p*d) to balance weight, processing time, and deadline urgency
    return f["dens"]


def choose_machine(i, f, c, M):
    # Assign to the machine that maximizes the minimum slack after adding job i
    # This preserves future flexibility for other jobs
    return M["slack_after"]
