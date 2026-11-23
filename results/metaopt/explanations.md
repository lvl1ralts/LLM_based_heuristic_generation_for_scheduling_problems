# Adversarial instances: explanations


## bimodal p, built against WSPT-Laxity

n=6, m=2, total weight 57. WSPT-Laxity loses 27, the optimum loses 20 (gap 18.92%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 100 | 113 | 10 | no | no |
| 1 | 100 | 130 | 10 | no | no |
| 2 | 100 | 146 | 10 | yes | yes |
| 3 | 85 | 162 | 10 | yes | yes |
| 4 | 11 | 178 | 10 | yes | yes |
| 5 | 100 | 194 | 7 | no | yes |

The optimum keeps this job set by giving job 5 (p=100, d=194, w=7, priority 0.104) its own room: it runs on machine 2 from 85 to 185, sharing it only with job(s) [3]. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 5 is 96, and it would finish at 196, after its deadline 194. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 5 (p=100, d=194, w=7, priority 0.104).


## bimodal p, built against WSPT

n=6, m=2, total weight 27. WSPT loses 18, the optimum loses 4 (gap 60.87%). Cause: **ranking**. Pattern: blocker holds capacity before the victim's deadline, low-value job ranked ahead of high-value job, ranking, several jobs traded.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 11 | 85 | 3 | yes | no |
| 1 | 8 | 86 | 1 | yes | yes |
| 2 | 4 | 87 | 5 | yes | yes |
| 3 | 85 | 88 | 1 | no | no |
| 4 | 85 | 93 | 10 | no | yes |
| 5 | 85 | 94 | 7 | no | yes |

WSPT ranks job 0 (p=11, d=85, w=3, priority 0.273) ahead of job 4 (p=85, d=93, w=10, priority 0.118). When 4's turn comes, admitting 4 on top of the jobs already accepted would make job 4 finish at 96, after its deadline 93, so 4 is rejected. The optimum makes the opposite choice: it drops 0 and keeps 4, worth 10 instead of 3.

WSPT ranks job 0 (p=11, d=85, w=3, priority 0.273) ahead of job 5 (p=85, d=94, w=7, priority 0.082). When 5's turn comes, admitting 5 on top of the jobs already accepted would make job 5 finish at 96, after its deadline 94, so 5 is rejected. The optimum makes the opposite choice: it drops 0 and keeps 5, worth 7 instead of 3.


## in-dist, built against WSPT-Laxity

n=6, m=2, total weight 54. WSPT-Laxity loses 17, the optimum loses 10 (gap 15.91%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 60 | 112 | 7 | yes | yes |
| 1 | 40 | 130 | 10 | yes | yes |
| 2 | 100 | 147 | 10 | no | no |
| 3 | 100 | 165 | 10 | yes | yes |
| 4 | 60 | 183 | 10 | yes | yes |
| 5 | 100 | 201 | 7 | no | yes |

The optimum keeps this job set by giving job 5 (p=100, d=201, w=7, priority 0.105) its own room: it runs on machine 2 from 100 to 200, sharing it only with job(s) [1, 4]. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 5 is 120, and it would finish at 220, after its deadline 201. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 5 (p=100, d=201, w=7, priority 0.105).


## in-dist, built against WSPT

n=6, m=2, total weight 45. WSPT loses 17, the optimum loses 7 (gap 26.32%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 5 | 40 | 7 | yes | yes |
| 1 | 20 | 41 | 10 | yes | yes |
| 2 | 40 | 42 | 7 | no | no |
| 3 | 5 | 43 | 10 | yes | yes |
| 4 | 5 | 44 | 1 | yes | yes |
| 5 | 40 | 45 | 10 | no | yes |

The optimum keeps this job set by giving job 5 (p=40, d=45, w=10, priority 0.250) its own room: it runs on machine 1 from 5 to 45, sharing it only with job(s) [0]. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 5 is 15, and it would finish at 55, after its deadline 45. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 5 (p=40, d=45, w=10, priority 0.250).


## large n, built against WSPT-Laxity

n=10, m=1, total weight 61. WSPT-Laxity loses 11, the optimum loses 11 (gap 0.00%). Cause: **ranking**. Pattern: ranking.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 100 | 168 | 1 | no | no |
| 1 | 80 | 189 | 1 | no | no |
| 2 | 100 | 230 | 1 | no | no |
| 3 | 60 | 250 | 1 | no | no |
| 4 | 100 | 271 | 7 | no | no |
| 5 | 100 | 291 | 10 | yes | yes |
| 6 | 100 | 312 | 10 | yes | yes |
| 7 | 80 | 332 | 10 | yes | yes |
| 8 | 100 | 598 | 10 | yes | yes |
| 9 | 20 | 639 | 10 | yes | yes |


## large n, built against WSPT

n=10, m=1, total weight 61. WSPT loses 30, the optimum loses 21 (gap 22.50%). Cause: **ranking**. Pattern: low-value job ranked ahead of high-value job, ranking.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 40 | 107 | 10 | yes | yes |
| 1 | 100 | 108 | 3 | no | no |
| 2 | 80 | 109 | 5 | no | no |
| 3 | 100 | 110 | 7 | no | no |
| 4 | 20 | 111 | 10 | yes | yes |
| 5 | 5 | 112 | 3 | yes | yes |
| 6 | 60 | 125 | 10 | no | yes |
| 7 | 5 | 126 | 1 | yes | no |
| 8 | 100 | 154 | 5 | no | no |
| 9 | 5 | 159 | 7 | yes | yes |

WSPT ranks job 7 (p=5, d=126, w=1, priority 0.200) ahead of job 6 (p=60, d=125, w=10, priority 0.167). When 6's turn comes, admitting 6 on top of the jobs already accepted would make job 7 finish at 130, after its deadline 126, so 6 is rejected. The optimum makes the opposite choice: it drops 7 and keeps 6, worth 10 instead of 1.


## lognormal p, built against WSPT-Laxity

n=6, m=2, total weight 53. WSPT-Laxity loses 33, the optimum loses 33 (gap 0.00%). Cause: **ranking**. Pattern: ranking.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 300 | 339 | 3 | no | no |
| 1 | 300 | 386 | 10 | no | no |
| 2 | 300 | 434 | 10 | no | no |
| 3 | 300 | 482 | 10 | no | yes |
| 4 | 300 | 530 | 10 | yes | no |
| 5 | 300 | 578 | 10 | yes | yes |

WSPT-Laxity ranks job 4 (p=300, d=530, w=10, priority 0.048) ahead of job 3 (p=300, d=482, w=10, priority 0.046). When 3's turn comes, admitting 3 on top of the jobs already accepted would make job 5 finish at 600, after its deadline 578, so 3 is rejected. The optimum makes the opposite choice: it drops 4 and keeps 3, worth 10 instead of 10.


## lognormal p, built against WSPT

n=6, m=2, total weight 37. WSPT loses 11, the optimum loses 1 (gap 27.78%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 3 | 120 | 1 | yes | yes |
| 1 | 50 | 121 | 10 | yes | yes |
| 2 | 3 | 122 | 5 | yes | yes |
| 3 | 20 | 123 | 10 | yes | yes |
| 4 | 120 | 124 | 1 | no | no |
| 5 | 120 | 125 | 10 | no | yes |

The optimum keeps this job set by giving job 5 (p=120, d=125, w=10, priority 0.083) its own room: it runs on machine 1 from 3 to 123, sharing it only with job(s) [2]. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 5 is 26, and it would finish at 146, after its deadline 125. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 5 (p=120, d=125, w=10, priority 0.083).


## loose T=0.1, built against WSPT-Laxity

n=6, m=2, total weight 53. WSPT-Laxity loses 10, the optimum loses 0 (gap 18.87%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 5 | 124 | 10 | yes | yes |
| 1 | 100 | 147 | 10 | no | yes |
| 2 | 100 | 171 | 10 | yes | yes |
| 3 | 60 | 195 | 10 | yes | yes |
| 4 | 40 | 219 | 3 | yes | yes |
| 5 | 100 | 242 | 10 | yes | yes |

The optimum keeps this job set by giving job 5 (p=100, d=242, w=10, priority 0.159) its own room: it runs on machine 1 from 100 to 200, sharing it only with job(s) [2]. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 5 is 145, and it would finish at 245, after its deadline 242. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 1 (p=100, d=147, w=10, priority 0.132).


## loose T=0.1, built against WSPT

n=6, m=2, total weight 36. WSPT loses 10, the optimum loses 0 (gap 27.78%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 5 | 20 | 7 | yes | yes |
| 1 | 5 | 21 | 5 | yes | yes |
| 2 | 20 | 22 | 10 | no | yes |
| 3 | 5 | 23 | 3 | yes | yes |
| 4 | 5 | 26 | 1 | yes | yes |
| 5 | 5 | 27 | 10 | yes | yes |

The optimum keeps this job set by giving job 2 (p=20, d=22, w=10, priority 0.500) its own room: it runs on machine 1 from 0 to 20, sharing it only with job(s) [5]. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 2 is 5, and it would finish at 25, after its deadline 22. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 2 (p=20, d=22, w=10, priority 0.500).


## many machines, built against WSPT-Laxity

n=6, m=3, total weight 33. WSPT-Laxity loses 9, the optimum loses 9 (gap 0.00%). Cause: **ranking**. Pattern: ranking.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 40 | 40 | 7 | no | yes |
| 1 | 40 | 45 | 10 | yes | yes |
| 2 | 40 | 48 | 7 | yes | no |
| 3 | 40 | 50 | 7 | yes | yes |
| 4 | 40 | 56 | 1 | no | no |
| 5 | 40 | 64 | 1 | no | no |

WSPT-Laxity ranks job 2 (p=40, d=48, w=7, priority 0.204) ahead of job 0 (p=40, d=40, w=7, priority 0.175). When 0's turn comes, admitting 0 on top of the jobs already accepted would make job 3 finish at 80, after its deadline 50, so 0 is rejected. The optimum makes the opposite choice: it drops 2 and keeps 0, worth 7 instead of 7.


## many machines, built against WSPT

n=6, m=3, total weight 40. WSPT loses 13, the optimum loses 5 (gap 22.86%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 20 | 80 | 1 | no | yes |
| 1 | 40 | 81 | 10 | yes | yes |
| 2 | 80 | 82 | 7 | no | yes |
| 3 | 80 | 83 | 5 | no | no |
| 4 | 20 | 84 | 7 | yes | yes |
| 5 | 80 | 85 | 10 | yes | yes |

The optimum keeps this job set by giving job 5 (p=80, d=85, w=10, priority 0.125) its own room: it runs on machine 2 from 0 to 80, alone. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 5 is 40, and it would finish at 120, after its deadline 85. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 0 (p=20, d=80, w=1, priority 0.050); job 2 (p=80, d=82, w=7, priority 0.087).


## narrow R=0.1, built against WSPT-Laxity

n=6, m=2, total weight 38. WSPT-Laxity loses 24, the optimum loses 18 (gap 30.00%). Cause: **ranking**. Pattern: blocker holds capacity before the victim's deadline, low-value job ranked ahead of high-value job, ranking, several jobs traded.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 40 | 63 | 3 | no | no |
| 1 | 60 | 65 | 10 | no | yes |
| 2 | 40 | 68 | 1 | no | no |
| 3 | 40 | 71 | 7 | yes | no |
| 4 | 40 | 74 | 7 | yes | no |
| 5 | 60 | 76 | 10 | no | yes |

WSPT-Laxity ranks job 3 (p=40, d=71, w=7, priority 0.251) ahead of job 1 (p=60, d=65, w=10, priority 0.179). When 1's turn comes, admitting 1 on top of the jobs already accepted would make job 4 finish at 80, after its deadline 74, so 1 is rejected. The optimum makes the opposite choice: it drops 3 and keeps 1, worth 10 instead of 7.

WSPT-Laxity ranks job 3 (p=40, d=71, w=7, priority 0.251) ahead of job 5 (p=60, d=76, w=10, priority 0.202). When 5's turn comes, admitting 5 on top of the jobs already accepted would make job 5 finish at 100, after its deadline 76, so 5 is rejected. The optimum makes the opposite choice: it drops 3 and keeps 5, worth 10 instead of 7.


## narrow R=0.1, built against WSPT

n=6, m=2, total weight 34. WSPT loses 14, the optimum loses 4 (gap 33.33%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 20 | 40 | 10 | yes | yes |
| 1 | 20 | 41 | 7 | yes | yes |
| 2 | 40 | 42 | 3 | no | no |
| 3 | 5 | 43 | 3 | yes | yes |
| 4 | 40 | 44 | 1 | no | no |
| 5 | 40 | 45 | 10 | no | yes |

The optimum keeps this job set by giving job 5 (p=40, d=45, w=10, priority 0.250) its own room: it runs on machine 1 from 5 to 45, sharing it only with job(s) [3]. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 5 is 20, and it would finish at 60, after its deadline 45. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 5 (p=40, d=45, w=10, priority 0.250).


## scale-shift, built against WSPT-Laxity

n=8, m=2, total weight 49. WSPT-Laxity loses 15, the optimum loses 13 (gap 5.56%). Cause: **ranking**. Pattern: blocker holds capacity before the victim's deadline, low-value job ranked ahead of high-value job, ranking.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 20 | 111 | 1 | yes | yes |
| 1 | 100 | 132 | 5 | no | no |
| 2 | 100 | 153 | 10 | yes | yes |
| 3 | 100 | 174 | 10 | yes | yes |
| 4 | 60 | 195 | 3 | yes | no |
| 5 | 100 | 216 | 5 | no | no |
| 6 | 100 | 237 | 5 | no | yes |
| 7 | 100 | 259 | 10 | yes | yes |

WSPT-Laxity ranks job 4 (p=60, d=195, w=3, priority 0.085) ahead of job 6 (p=100, d=237, w=5, priority 0.079). When 6's turn comes, admitting 6 on top of the jobs already accepted would make job 7 finish at 260, after its deadline 259, so 6 is rejected. The optimum makes the opposite choice: it drops 4 and keeps 6, worth 5 instead of 3.


## scale-shift, built against WSPT

n=8, m=2, total weight 43. WSPT loses 6, the optimum loses 5 (gap 2.63%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 100 | 100 | 7 | yes | yes |
| 1 | 20 | 102 | 7 | yes | yes |
| 2 | 20 | 104 | 10 | yes | yes |
| 3 | 5 | 105 | 5 | yes | yes |
| 4 | 5 | 106 | 3 | yes | yes |
| 5 | 40 | 107 | 1 | no | yes |
| 6 | 20 | 108 | 5 | yes | yes |
| 7 | 100 | 124 | 5 | no | no |

The optimum keeps this job set by giving job 6 (p=20, d=108, w=5, priority 0.250) its own room: it runs on machine 1 from 85 to 105, sharing it only with job(s) [1, 2, 4, 5]. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 6 is 90, and it would finish at 110, after its deadline 108. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 5 (p=40, d=107, w=1, priority 0.025).


## single m=1, built against WSPT-Laxity

n=8, m=1, total weight 52. WSPT-Laxity loses 11, the optimum loses 5 (gap 12.77%). Cause: **ranking**. Pattern: blocker holds capacity before the victim's deadline, low-value job ranked ahead of high-value job, ranking, several jobs traded.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 60 | 132 | 7 | yes | yes |
| 1 | 5 | 175 | 3 | yes | no |
| 2 | 100 | 217 | 10 | no | yes |
| 3 | 100 | 260 | 10 | yes | yes |
| 4 | 100 | 302 | 1 | no | no |
| 5 | 80 | 345 | 10 | yes | yes |
| 6 | 40 | 387 | 10 | yes | yes |
| 7 | 60 | 430 | 1 | yes | no |

WSPT-Laxity ranks job 1 (p=5, d=175, w=3, priority 1.183) ahead of job 2 (p=100, d=217, w=10, priority 0.154). When 2's turn comes, admitting 2 on top of the jobs already accepted would make job 3 finish at 265, after its deadline 260, so 2 is rejected. The optimum makes the opposite choice: it drops 1 and keeps 2, worth 10 instead of 3.


## single m=1, built against WSPT

n=8, m=1, total weight 46. WSPT loses 20, the optimum loses 11 (gap 25.71%). Cause: **ranking**. Pattern: low-value job ranked ahead of high-value job, ranking.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 60 | 80 | 10 | no | yes |
| 1 | 80 | 81 | 7 | no | no |
| 2 | 5 | 82 | 1 | yes | no |
| 3 | 5 | 83 | 3 | yes | yes |
| 4 | 5 | 84 | 5 | yes | yes |
| 5 | 5 | 86 | 7 | yes | yes |
| 6 | 20 | 95 | 10 | yes | yes |
| 7 | 60 | 96 | 3 | no | no |

WSPT ranks job 2 (p=5, d=82, w=1, priority 0.200) ahead of job 0 (p=60, d=80, w=10, priority 0.167). When 0's turn comes, admitting 0 on top of the jobs already accepted would make job 6 finish at 100, after its deadline 95, so 0 is rejected. The optimum makes the opposite choice: it drops 2 and keeps 0, worth 10 instead of 1.


## tight T=0.8, built against WSPT-Laxity

n=8, m=1, total weight 52. WSPT-Laxity loses 32, the optimum loses 28 (gap 16.67%). Cause: **ranking**. Pattern: blocker holds capacity before the victim's deadline, ranking, several jobs traded.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 60 | 116 | 1 | no | no |
| 1 | 100 | 134 | 7 | no | no |
| 2 | 100 | 153 | 3 | no | no |
| 3 | 60 | 171 | 10 | yes | yes |
| 4 | 80 | 190 | 7 | no | yes |
| 5 | 100 | 208 | 10 | yes | no |
| 6 | 100 | 227 | 7 | no | no |
| 7 | 100 | 245 | 7 | no | yes |

WSPT-Laxity ranks job 5 (p=100, d=208, w=10, priority 0.152) ahead of job 4 (p=80, d=190, w=7, priority 0.138). When 4's turn comes, admitting 4 on top of the jobs already accepted would make job 5 finish at 240, after its deadline 208, so 4 is rejected. The optimum makes the opposite choice: it drops 5 and keeps 4, worth 7 instead of 10.

WSPT-Laxity ranks job 5 (p=100, d=208, w=10, priority 0.152) ahead of job 7 (p=100, d=245, w=7, priority 0.111). When 7's turn comes, admitting 7 on top of the jobs already accepted would make job 7 finish at 260, after its deadline 245, so 7 is rejected. The optimum makes the opposite choice: it drops 5 and keeps 7, worth 7 instead of 10.


## tight T=0.8, built against WSPT

n=8, m=1, total weight 44. WSPT loses 19, the optimum loses 10 (gap 26.47%). Cause: **ranking**. Pattern: low-value job ranked ahead of high-value job, ranking.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 80 | 80 | 1 | no | no |
| 1 | 80 | 81 | 5 | no | no |
| 2 | 60 | 82 | 10 | no | yes |
| 3 | 5 | 83 | 7 | yes | yes |
| 4 | 60 | 84 | 3 | no | no |
| 5 | 40 | 105 | 7 | yes | yes |
| 6 | 5 | 106 | 1 | yes | no |
| 7 | 5 | 114 | 10 | yes | yes |

WSPT ranks job 6 (p=5, d=106, w=1, priority 0.200) ahead of job 2 (p=60, d=82, w=10, priority 0.167). When 2's turn comes, admitting 2 on top of the jobs already accepted would make job 6 finish at 110, after its deadline 106, so 2 is rejected. The optimum makes the opposite choice: it drops 6 and keeps 2, worth 10 instead of 1.


## w anticorr, built against WSPT-Laxity

n=6, m=2, total weight 33. WSPT-Laxity loses 9, the optimum loses 5 (gap 14.29%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 100 | 100 | 4 | no | yes |
| 1 | 100 | 116 | 4 | no | no |
| 2 | 20 | 131 | 12 | yes | yes |
| 3 | 100 | 147 | 1 | no | no |
| 4 | 80 | 163 | 6 | yes | yes |
| 5 | 80 | 179 | 6 | yes | yes |

The optimum keeps this job set by giving job 5 (p=80, d=179, w=6, priority 0.116) its own room: it runs on machine 2 from 80 to 160, sharing it only with job(s) [4]. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 5 is 100, and it would finish at 180, after its deadline 179. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 0 (p=100, d=100, w=4, priority 0.040).


## w anticorr, built against WSPT

n=6, m=2, total weight 63. WSPT loses 17, the optimum loses 7 (gap 17.86%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 5 | 40 | 14 | yes | yes |
| 1 | 20 | 41 | 9 | yes | yes |
| 2 | 5 | 42 | 11 | yes | yes |
| 3 | 40 | 43 | 7 | no | no |
| 4 | 20 | 45 | 12 | yes | yes |
| 5 | 40 | 46 | 10 | no | yes |

The optimum keeps this job set by giving job 5 (p=40, d=46, w=10, priority 0.250) its own room: it runs on machine 1 from 5 to 45, sharing it only with job(s) [2]. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 5 is 20, and it would finish at 60, after its deadline 46. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 5 (p=40, d=46, w=10, priority 0.250).


## w correlated, built against WSPT-Laxity

n=6, m=2, total weight 64. WSPT-Laxity loses 39, the optimum loses 23 (gap 39.02%). Cause: **ranking**. Pattern: blocker holds capacity before the victim's deadline, low-value job ranked ahead of high-value job, ranking, several jobs traded.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 20 | 100 | 3 | yes | yes |
| 1 | 100 | 116 | 11 | no | no |
| 2 | 80 | 131 | 12 | yes | no |
| 3 | 60 | 147 | 10 | yes | yes |
| 4 | 100 | 163 | 14 | no | yes |
| 5 | 100 | 179 | 14 | no | yes |

WSPT-Laxity ranks job 2 (p=80, d=131, w=12, priority 0.208) ahead of job 4 (p=100, d=163, w=14, priority 0.194). When 4's turn comes, admitting 4 on top of the jobs already accepted would make job 4 finish at 180, after its deadline 163, so 4 is rejected. The optimum makes the opposite choice: it drops 2 and keeps 4, worth 14 instead of 12.

WSPT-Laxity ranks job 2 (p=80, d=131, w=12, priority 0.208) ahead of job 5 (p=100, d=179, w=14, priority 0.202). When 5's turn comes, admitting 5 on top of the jobs already accepted would make job 5 finish at 180, after its deadline 179, so 5 is rejected. The optimum makes the opposite choice: it drops 2 and keeps 5, worth 14 instead of 12.


## w correlated, built against WSPT

n=6, m=2, total weight 59. WSPT loses 37, the optimum loses 21 (gap 42.11%). Cause: **decoder**. Pattern: decoder, earliest-free placement fragments machine capacity, long job must sit alone on a machine.

| job | p | d | w | rule keeps | optimum keeps |
|---|---|---|---|---|---|
| 0 | 80 | 100 | 12 | no | yes |
| 1 | 5 | 101 | 4 | yes | yes |
| 2 | 60 | 102 | 10 | yes | no |
| 3 | 40 | 103 | 8 | yes | yes |
| 4 | 100 | 104 | 11 | no | no |
| 5 | 100 | 140 | 14 | no | yes |

The optimum keeps this job set by giving job 5 (p=100, d=140, w=14, priority 0.140) its own room: it runs on machine 2 from 40 to 140, sharing it only with job(s) [3]. The decoder's test instead places every job, in deadline order, on the machine that becomes free first. That spreads the earlier jobs over all machines, so the earliest any machine is free for job 5 is 45, and it would finish at 145, after its deadline 140. The test rejects the set, so no priority order can make the decoder keep it.

Jobs lost as a result: job 0 (p=80, d=100, w=12, priority 0.150); job 5 (p=100, d=140, w=14, priority 0.140). Kept instead: job 2 (p=60, d=102, w=10, priority 0.167).
