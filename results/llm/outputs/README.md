# LLM-generated heuristics: outputs (report 14)
One folder per model. A heuristic is one rule, written once by the LLM and then used unchanged on every instance.
- `heuristic.py`: the heuristic that was tested, with the model's own idea
- `all_heuristics.md`: every heuristic the model wrote in its run
- `per_instance.csv`: weight lost on every instance, next to WSPT, the best of the 17 rules on that instance, the WSPT seed and GA-joint
Top level: `per_instance.csv` (all methods), `summary_by_set.csv`.

## Mean gap (%) per instance set
Family: gap to the LP bound, averaged over the 12 families as in report 01. GA-joint was run on only 24 of the 102 family instances (2 per family), so the second column repeats the comparison on those 24. Adversarial and similarity: gap to the exact optimum.

| Method | family (102) | family, GA's 24 (24) | adversarial (24) | similarity m=1 (40) | similarity m=4 (40) |
|---|---|---|---|---|---|
| WSPT | 2.04 | 1.95 | 20.62 | 0.31 | 2.90 |
| best of 17 rules (per instance) | 1.91 | 1.84 | 8.21 | 0.22 | 2.31 |
| WSPT seed, new decoder | 1.65 | 1.82 | 13.55 | 0.31 | 2.85 |
| **Gemini 3.8 Flash** | 1.66 | 1.83 | 12.93 | 0.37 | 2.76 |
| **Gemma 4 31B** | 1.63 | 1.79 | 13.55 | 0.31 | 2.72 |
| **GPT-OSS 120B** | 2.20 | 2.28 | 11.82 | 1.31 | 4.17 |
| **Qwen 3.8 27B** | 3.33 | 3.05 | 6.93 | 4.27 | 4.11 |
| GA-joint | 0.93 | 0.93 | 0.00 | 0.01 | 0.38 |

## Models

| Folder | Called through | Written | Ran | Tested heuristic's idea |
|---|---|---|---|---|
| `Gemini_3.8_Flash/` | Google Gemini API (Gemini key) | 6 | 6 | Prioritize jobs primarily by weight density with a gentle boost for earlier deadline rank, and place each job on the feasible machine with the highest current load to pack utilized machines and preserve empty ones. |
| `Gemma_4_31B/` | Google Gemini API (Gemini key); open-weights model | 24 | 21 | Use WSPT priority to focus on high-value jobs and a Best-Fit machine selection strategy by maximizing the load on the chosen machine to preserve large contiguous time blocks on other machines. |
| `GPT-OSS_120B/` | Groq API (Groq key); OpenAI open-weights model | 18 | 16 | Improve priority by combining weight, weight‑per‑processing, tight headroom and urgency, and balance machines by also considering current load, remaining slack and number of jobs on each machine. |
| `Qwen_3.8_27B/` | Groq API (Groq key); Alibaba open-weights model | 18 | 9 | Maximize the urgency-adjusted value of jobs (weight scaled by laxity to favor early deadlines with high value) and assign them to the machine that minimizes the absolute difference between its current load and the ideal balanced load to ensure even distribution while preserving feasibility via tightest slack consideration as a secondary tie-breaker. |
