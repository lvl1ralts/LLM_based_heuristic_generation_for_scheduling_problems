"""GPT-OSS 120B: the heuristic tested in report 14.

Called through: Groq API (Groq key); OpenAI open-weights model
Written in generation 3 by operator m1 (heuristic #17 of the run).
Training gap 3.921% (WSPT seed: 3.503%, WSPT: 4.183%).

The model's own idea:
  Improve priority by combining weight, weight‑per‑processing, tight headroom and urgency, and balance machines by also considering current load, remaining slack and number of jobs on each machine.

priority(f, c): one score per job, the jobs are tried highest first.
choose_machine(i, f, c, M): one score per machine for job i; the job goes on
the machine that fits with the highest score, or is late if none fits.
"""
import math

import numpy as np


def priority(f, c):
    """
    High weight, high weight‑per‑processing, tight headroom and high urgency
    increase the score; short processing time further helps.
    """
    eps = 1e-9
    w      = f["w"]
    wp     = f["wp"]                 # w / p
    head   = f["headroom_norm"]      # 0 = tight, 1 = loose
    urg    = f["urgency"]            # p / d
    p_norm = f["p_norm"]             # normalised processing time

    # (1‑head) favours tight periods, urg favours early‑deadline jobs
    score = w * wp * (1.0 - head) * (1.0 + urg) / (p_norm + eps)
    return score


def choose_machine(i, f, c, M):
    """
    Among feasible machines, prefer the one that keeps total load low,
    leaves more slack, and balances the number of assigned jobs.
    """
    fits   = M["fits"]                # bool array
    load   = M["load"]                # current load per machine
    slack  = M["slack_after"]         # slack after inserting job i
    count  = M["count"]               # number of jobs already on the machine
    p_i    = f["p"][i]

    load_after = load + p_i

    # higher score = better machine
    # -load_after  : minimise load
    # +0.2*slack   : reward remaining slack
    # -0.5*count   : spread jobs evenly
    score = np.where(fits,
                     -load_after + 0.2 * slack - 0.5 * count,
                     -np.inf)
    return score
