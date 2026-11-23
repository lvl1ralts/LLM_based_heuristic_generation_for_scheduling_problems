"""Gemini 3.8 Flash: the heuristic tested in report 14.

Called through: Google Gemini API (Gemini key)
Written in generation 1 by operator e1 (heuristic #4 of the run).
Training gap 3.196% (WSPT seed: 3.503%, WSPT: 4.183%).

The model's own idea:
  Prioritize jobs primarily by weight density with a gentle boost for earlier deadline rank, and place each job on the feasible machine with the highest current load to pack utilized machines and preserve empty ones.

priority(f, c): one score per job, the jobs are tried highest first.
choose_machine(i, f, c, M): one score per machine for job i; the job goes on
the machine that fits with the highest score, or is late if none fits.
"""
import math

import numpy as np


def priority(f, c):
    return f["wp"] * (1.0 + 0.1 * (1.0 - f["d_rank"]))


def choose_machine(i, f, c, M):
    return M["load"] - 1e-4 * np.arange(len(M["load"]))
