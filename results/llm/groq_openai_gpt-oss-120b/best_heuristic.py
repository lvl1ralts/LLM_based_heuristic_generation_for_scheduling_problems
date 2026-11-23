"""groq_openai_gpt-oss-120b, generation 0 (seed), training gap 3.503%

WSPT priority (weight per unit of processing time); the job goes on the first machine where it fits.
"""
import math

import numpy as np


def priority(f, c):
    return f["wp"]


def choose_machine(i, f, c, M):
    return -np.arange(len(M["fits"]))
