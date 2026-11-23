"""gemini-3.7-flash, generation 0 (i1), training gap 3.363%

Prioritize jobs by weight-to-processing ratio boosted for urgent and early-deadline jobs, then place each job using a Best-Fit rule that favors busy machines while retaining feasible slack buffer.
"""
import math

import numpy as np


def priority(f, c):
    return (
        f["wp"]
        * (1.0 + f["urgency"])
        * (1.5 - 0.5 * f["d_rank"])
        / (1.0 + 0.1 * f["p_norm"])
    )


def choose_machine(i, f, c, M):
    return M["load"] + 0.05 * M["slack_after"]
