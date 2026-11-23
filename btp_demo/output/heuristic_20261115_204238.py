"""Heuristic for Pm | r=0 | sum w U  (live: groq:qwen/qwen3.8-27b, answered in 1s)

Idea: Combine WSPT priority with a machine selection strategy that prioritizes machines where the job's placement leaves the highest surplus time before the minimum deadline conflict, balancing immediate fit and future flexibility by using the difference between slack_after and the current machine load to favor machines with more headroom.
"""
import math

import numpy as np


def priority(f, c):
    return f["wp"]

def choose_machine(i, f, c, M):
    # Favor machines where adding job i leaves more relative slack compared to current load
    # Normalize slack_after by current load to account for different load levels
    loads = M["load"] + 1e-6  # Avoid division by zero
    slack_ratio = M["slack_after"] / loads
    return slack_ratio
