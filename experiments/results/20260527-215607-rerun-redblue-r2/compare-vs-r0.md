# Comparison: **pipeline-r2 (rerun-v2)** vs **pipeline-r0 (orig)**

Cohen's d is computed on paired per-question observations. Positive d means **B > A** on the metric. Conventional thresholds: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.

| metric | n | mean(A) | mean(B) | Δ | Cohen's d |
|---|---|---|---|---|---|
| factual_accuracy | 3 | 1.000 | 1.000 | +0.000 ⬇ worse | +nan |
| hallucination_rate | 3 | 0.169 | 0.153 | -0.016 ⬇ better | -0.476 |
| citation_coverage | 3 | 0.413 | 0.521 | +0.108 ⬆ better | +2.049 |
| judge_overall | 3 | 4.400 | 4.420 | +0.020 ⬆ better | +0.141 |
