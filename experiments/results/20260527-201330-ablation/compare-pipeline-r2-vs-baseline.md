# Comparison: **pipeline-r2** vs **baseline**

Cohen's d is computed on paired per-question observations. Positive d means **B > A** on the metric. Conventional thresholds: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.

| metric | n | mean(A) | mean(B) | Δ | Cohen's d |
|---|---|---|---|---|---|
| factual_accuracy | 35 | 0.960 | 0.996 | +0.036 ⬆ better | +0.520 |
| hallucination_rate | 35 | 0.146 | 0.156 | +0.010 ⬆ worse | +0.080 |
| citation_coverage | 35 | 0.000 | 0.320 | +0.320 ⬆ better | +2.626 |
| judge_overall | 35 | 4.240 | 4.285 | +0.044 ⬆ better | +0.143 |
