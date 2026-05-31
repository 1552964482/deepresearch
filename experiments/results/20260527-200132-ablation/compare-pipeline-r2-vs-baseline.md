# Comparison: **pipeline-r2** vs **baseline**

Cohen's d is computed on paired per-question observations. Positive d means **B > A** on the metric. Conventional thresholds: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.

| metric | n | mean(A) | mean(B) | Δ | Cohen's d |
|---|---|---|---|---|---|
| factual_accuracy | 3 | 0.667 | 1.000 | +0.333 ⬆ better | +0.816 |
| hallucination_rate | 3 | 0.107 | 0.117 | +0.010 ⬆ worse | +0.085 |
| citation_coverage | 3 | 0.000 | 0.335 | +0.335 ⬆ better | +8.206 |
| judge_overall | 2 | 4.395 | 4.615 | +0.220 ⬆ better | +0.830 |
