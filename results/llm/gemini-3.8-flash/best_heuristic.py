"""gemini-3.8-flash, generation 1 (e1), training gap 3.196%

Prioritize jobs primarily by weight density with a gentle boost for earlier deadline rank, and place each job on the feasible machine with the highest current load to pack utilized machines and preserve empty ones.
"""
import math

import numpy as np


def priority(f, c):
    return f["wp"] * (1.0 + 0.1 * (1.0 - f["d_rank"]))


def choose_machine(i, f, c, M):
    return M["load"] - 1e-4 * np.arange(len(M["load"]))
