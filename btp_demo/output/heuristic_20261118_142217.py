"""Heuristic for Pm | r=0 | sum w U  (saved: Gemma 4 31B (report))

Idea: Use WSPT priority to focus on high-value jobs and a Best-Fit machine selection strategy by maximizing the load on the chosen machine to preserve large contiguous time blocks on other machines.
"""
import math

import numpy as np


def priority(f, c):
    return f["wp"]


def choose_machine(i, f, c, M):
    return M["load"]
