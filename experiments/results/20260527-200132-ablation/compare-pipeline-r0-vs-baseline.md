# Comparison: **pipeline-r0** vs **baseline**

Cohen's d is computed on paired per-question observations. Positive d means **B > A** on the metric. Conventional thresholds: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.

| metric | n | mean(A) | mean(B) | Δ | Cohen's d |
|---|---|---|---|---|---|
| factual_accuracy | 3 | 0.667 | 1.000 | +0.333 ⬆ better | +0.816 |
| hallucination_rate | 3 | 0.107 | 0.109 | +0.002 ⬆ worse | +0.021 |
| citation_coverage | 3 | 0.000 | 0.356 | +0.356 ⬆ better | +108.671 |
| judge_overall | 2 | 4.395 | 4.760 | +0.365 ⬆ better | +1.272 |
