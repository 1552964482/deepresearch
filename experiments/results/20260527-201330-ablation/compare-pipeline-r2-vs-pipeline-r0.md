# Comparison: **pipeline-r2** vs **pipeline-r0**

Cohen's d is computed on paired per-question observations. Positive d means **B > A** on the metric. Conventional thresholds: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.

| metric | n | mean(A) | mean(B) | Δ | Cohen's d |
|---|---|---|---|---|---|
| factual_accuracy | 35 | 1.000 | 0.996 | -0.004 ⬇ worse | -0.239 |
| hallucination_rate | 35 | 0.164 | 0.156 | -0.008 ⬇ better | -0.082 |
| citation_coverage | 35 | 0.393 | 0.320 | -0.073 ⬇ worse | -0.561 |
| judge_overall | 35 | 4.299 | 4.285 | -0.014 ⬇ worse | -0.032 |
