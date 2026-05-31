# Comparison: **pipeline-r2** vs **pipeline-r0**

Cohen's d is computed on paired per-question observations. Positive d means **B > A** on the metric. Conventional thresholds: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.

| metric | n | mean(A) | mean(B) | Δ | Cohen's d |
|---|---|---|---|---|---|
| factual_accuracy | 3 | 1.000 | 1.000 | +0.000 ⬇ worse | +nan |
| hallucination_rate | 3 | 0.109 | 0.117 | +0.007 ⬆ worse | +0.073 |
| citation_coverage | 3 | 0.356 | 0.335 | -0.021 ⬇ worse | -0.512 |
| judge_overall | 3 | 4.790 | 4.577 | -0.213 ⬇ worse | -2.175 |
