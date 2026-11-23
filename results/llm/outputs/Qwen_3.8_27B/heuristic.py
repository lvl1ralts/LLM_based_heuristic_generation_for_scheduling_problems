"""Qwen 3.8 27B: the heuristic tested in report 14.

Called through: Groq API (Groq key); Alibaba open-weights model
Written in generation 2 by operator e1 (heuristic #17 of the run).
Training gap 3.809% (WSPT seed: 3.503%, WSPT: 4.183%).

The model's own idea:
  Maximize the urgency-adjusted value of jobs (weight scaled by laxity to favor early deadlines with high value) and assign them to the machine that minimizes the absolute difference between its current load and the ideal balanced load to ensure even distribution while preserving feasibility via tightest slack consideration as a secondary tie-breaker.

priority(f, c): one score per job, the jobs are tried highest first.
choose_machine(i, f, c, M): one score per machine for job i; the job goes on
the machine that fits with the highest score, or is late if none fits.
"""
import math

import numpy as np


def priority(f, c):
    w = f["w"]
    laxity = f["laxity"]
    # High weight and high laxity (early deadline relative to length) is good
    # Laxity=(d-p)/d. High laxity means p is small relative to d, but we also want early deadlines?
    # Actually, let's look at the provided heuristic 2 which used w/p/d.
    # New idea: Combine weight and urgency. Let's use w * laxity? 
    # Or perhaps: maximize w / d (value per unit time) but penalize long jobs more strongly?
    # Let's try: Score = w * (1/d) - alpha * p_norm? 
    # The prompt asks for a clearly different idea.
    # Heuristic 1: Density (w/p) minus small p penalty.
    # Heuristic 2: Ratio (w/p)/d.
    # Let's try: Weight scaled by inverse deadline, but also adjusting for slack availability globally?
    # Simpler different idea: Rank by strict Earliest Deadline First weighted by value?
    # Score = w / d. This is different from w/p/d and w/p.
    # Let's add a slight premium for jobs that are "tight" to process (small p) but weighted by value.
    # Actually, let's stick to: Score = w / (d * p_norm_safe) is close to h2.
    
    # Distinct Idea: Prioritize jobs with high weight that have the most "margin" in the schedule 
    # globally, or simply: Maximize (w - p) ? No.
    # Let's use: Score = w * (1 - p/max_p) + w / d ?
    
    # Let's go with: Score = w / d (Value over Deadline).
    # This rewards high weight jobs with earlier deadlines.
    # It does not explicitly divide by processing time p in the numerator/denominator ratio like h1/h2.
    d_safe = np.where(f["d"] == 0, 1, f["d"])
    score = f["w"] / d_safe
    return score

def choose_machine(i, f, c, M):
    fits = M["fits"]
    load = M["load"]
    total_p = c["total_p"]
    m = c["m"]
    # Ideal load per machine
    ideal_load = total_p / m
    
    # If we only knew total_p of feasible jobs, we could be more precise, 
    # but total_p of all jobs is a reasonable proxy for the target balance.
    
    # We want the machine that is closest to its ideal load, OR has the most slack.
    # The prompt says: "Job goes on the feasible machine with the highest score."
    # Let's score based on how well adding this job keeps the load near balance.
    # If machine is under-loaded, it's good. If over-loaded, it's bad.
    # Score = -abs(load - ideal_load) for feasible machines?
    # But we must also ensure feasibility is handled. The problem statement says 
    # "Job i is offered only to the machines where it still fits".
    # So `fits` is already true for candidates.
    
    # However, simply balancing load might cause tight deadlines to be missed if 
    # the machine with less load has later deadlines? 
    # No, the `fits` check ensures all jobs (including the new one) stay on time 
    # *if* run in deadline order.
    
    # So, among feasible machines, let's pick the one that keeps the load most even.
    # Score is high when load is close to ideal.
    # distance = abs(load - ideal_load)
    # score = -distance
    
    # But wait, standard load balancing often prefers the machine with the *least* load 
    # to be safe, or the one closest to average. 
    # Let's prefer the machine with the lowest load among feasible ones?
    # If we pick lowest load, it's simple.
    # Is "lowest load" clearly different from h1 and h2?
    # h1: Highest slack_after.
    # h2: Highest slack_after.
    # Both h1 and h2 used slack_after primarily.
    
    # So if we use Load-based balancing, it is clearly different.
    # Let's use: Score = -load for feasible machines. (Minimize load)
    # This distributes work evenly.
    
    # To be more sophisticated, we can combine load balancing with a tiny bit of slack consideration?
    # No, keep it simple and distinct.
    # Strategy: Assign to the feasible machine with the currently lowest total load.
    
    # To implement "pick lowest load", we return scores such that argmax picks the lowest load.
    # score = -load
    
    # We must mask infeasible machines.
    scores = np.where(fits, -load, -np.inf)
    
    return scores
