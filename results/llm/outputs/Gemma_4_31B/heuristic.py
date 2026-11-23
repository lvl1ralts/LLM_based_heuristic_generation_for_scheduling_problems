"""Gemma 4 31B: the heuristic tested in report 14.

Called through: Google Gemini API (Gemini key); open-weights model
Written in generation 3 by operator m1 (heuristic #16 of the run).
Training gap 3.470% (WSPT seed: 3.503%, WSPT: 4.183%).

The model's own idea:
  Use WSPT priority to focus on high-value jobs and a Best-Fit machine selection strategy by maximizing the load on the chosen machine to preserve large contiguous time blocks on other machines.

priority(f, c): one score per job, the jobs are tried highest first.
choose_machine(i, f, c, M): one score per machine for job i; the job goes on
the machine that fits with the highest score, or is late if none fits.
"""
import math

import numpy as np


def priority(f, c):
    return f["wp"]


def choose_machine(i, f, c, M):
    return M["load"]
