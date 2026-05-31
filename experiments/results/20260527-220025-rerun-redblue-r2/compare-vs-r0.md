# Comparison: **pipeline-r2 (rerun-v2)** vs **pipeline-r0 (orig)**

Cohen's d is computed on paired per-question observations. Positive d means **B > A** on the metric. Conventional thresholds: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.

| metric | n | mean(A) | mean(B) | Δ | Cohen's d |
|---|---|---|---|---|---|
| factual_accuracy | 35 | 1.000 | 1.000 | +0.000 ⬇ worse | +nan |
| hallucination_rate | 35 | 0.164 | 0.158 | -0.005 ⬇ better | -0.055 |
| citation_coverage | 35 | 0.393 | 0.472 | +0.079 ⬆ better | +1.312 |
| judge_overall | 35 | 4.299 | 4.329 | +0.030 ⬆ better | +0.060 |
