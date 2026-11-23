"""gemma-4-31b-it, generation 3 (m1), training gap 3.470%

Use WSPT priority to focus on high-value jobs and a Best-Fit machine selection strategy by maximizing the load on the chosen machine to preserve large contiguous time blocks on other machines.
"""
import math

import numpy as np


def priority(f, c):
    return f["wp"]


def choose_machine(i, f, c, M):
    return M["load"]
